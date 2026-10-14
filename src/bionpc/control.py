"""
Designer Controllability, Chemical State Management, and INI/JSON Profile Loader.

Addresses the "Black Box" problem in Game AI by giving game designers explicit,
interpretable knobs over the biological CTRNN:
- Lock or clamp recurrent neurotransmitter/APL inhibition states per layer
- Inject instantaneous chemical spikes (e.g., from an LLM Forebrain or scripted trigger)
- Scale or bias sensory channels without retraining weights
- Clamp or force specific motor action outputs (e.g., safe-zone peace rules)
- Load/save human-readable .ini and .json designer profiles
"""

from __future__ import annotations

import configparser
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

BIONPC_MAX_LAYERS = 16
BIONPC_MAX_DIM = 64
STATE_MAGIC = b"BNPCST01"


@dataclass
class AgentState:
    """
    Continuous biological recurrent state for one or more agents across game ticks.
    Matches C `bionpc_agent_state_t` layout for deterministic serialization.
    """
    apl_state: torch.Tensor  # shape: [batch_size, n_layers]
    apl_locked: Optional[torch.Tensor] = None  # bool mask shape: [batch_size, n_layers]

    @classmethod
    def zeros(
        cls,
        batch_size: int = 1,
        n_layers: int = 2,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AgentState":
        return cls(
            apl_state=torch.zeros(batch_size, n_layers, device=device, dtype=dtype),
            apl_locked=torch.zeros(batch_size, n_layers, device=device, dtype=torch.bool),
        )

    def clone(self) -> "AgentState":
        return AgentState(
            apl_state=self.apl_state.clone(),
            apl_locked=self.apl_locked.clone() if self.apl_locked is not None else None,
        )

    def spike(self, delta: Union[float, List[float], torch.Tensor], layer: Optional[int] = None) -> None:
        """
        Inject an instantaneous neurotransmitter/chemical spike into the recurrent state.
        Used by scripted triggers or the Hybrid LLM Forebrain.
        """
        if layer is not None:
            if self.apl_locked is not None:
                mask = ~self.apl_locked[:, layer]
                self.apl_state[mask, layer] += delta
            else:
                self.apl_state[:, layer] += delta
        else:
            delta_t = torch.as_tensor(delta, device=self.apl_state.device, dtype=self.apl_state.dtype)
            if self.apl_locked is not None:
                self.apl_state = torch.where(self.apl_locked, self.apl_state, self.apl_state + delta_t)
            else:
                self.apl_state = self.apl_state + delta_t

    def set_state(self, value: float, layer: int = 0, locked: Optional[bool] = None) -> None:
        """Set the chemical state for a specific layer and optionally lock/unlock its decay."""
        self.apl_state[:, layer] = value
        if locked is not None:
            if self.apl_locked is None:
                self.apl_locked = torch.zeros_like(self.apl_state, dtype=torch.bool)
            self.apl_locked[:, layer] = locked

    def crc32(self) -> int:
        """
        Compute deterministic IEEE-754 CRC32 checksum of the recurrent state buffer
        for lockstep multiplayer synchronization verification.
        """
        arr = self.apl_state.detach().cpu().numpy().astype("<f4")
        return zlib.crc32(arr.tobytes()) & 0xFFFFFFFF

    def to_bytes(self) -> bytes:
        """Serialize agent states to compact binary format matching C state snapshots."""
        arr = self.apl_state.detach().cpu().numpy().astype("<f4")
        batch_size, n_layers = arr.shape
        locked_arr = (
            self.apl_locked.detach().cpu().numpy().astype("<u1")
            if self.apl_locked is not None
            else np.zeros((batch_size, n_layers), dtype="<u1")
        )
        header = struct.pack("<8sii", STATE_MAGIC, batch_size, n_layers)
        return header + arr.tobytes() + locked_arr.tobytes()

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        device: Union[str, torch.device] = "cpu",
    ) -> "AgentState":
        """Deserialize agent states from compact binary snapshot."""
        if len(data) < 16:
            raise ValueError("Invalid state buffer: too short")
        magic, batch_size, n_layers = struct.unpack("<8sii", data[:16])
        if magic != STATE_MAGIC:
            raise ValueError(f"Invalid state magic: {magic!r}")
        float_bytes = batch_size * n_layers * 4
        lock_bytes = batch_size * n_layers
        arr = np.frombuffer(data[16 : 16 + float_bytes], dtype="<f4").reshape(batch_size, n_layers).copy()
        locked = (
            np.frombuffer(data[16 + float_bytes : 16 + float_bytes + lock_bytes], dtype="<u1")
            .reshape(batch_size, n_layers)
            .astype(bool)
            .copy()
        )
        return cls(
            apl_state=torch.from_numpy(arr).to(device),
            apl_locked=torch.from_numpy(locked).to(device),
        )

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "AgentState":
        return cls.from_bytes(Path(path).read_bytes(), device=device)


@dataclass
class DesignerControl:
    """
    Interpretable designer controls and guardrails applied during BioNPC inference.
    """
    # Sensory input modulation: s_eff = s * sensory_scale + sensory_bias
    sensory_scale: Optional[List[float]] = None
    sensory_bias: Optional[List[float]] = None

    # Chemical / APL recurrent state bounds & decay overrides
    apl_min: float = -10.0
    apl_max: float = 10.0
    apl_locked: bool = False
    chemical_bias: float = 0.0

    # Motor action output modulation and clamping: a_eff = clamp(a * action_scale, action_min, action_max)
    action_scale: Optional[List[float]] = None
    action_min: Optional[List[float]] = None
    action_max: Optional[List[float]] = None

    # Explicit designer action overrides: index -> forced float value (e.g. {1: 0.0} disables Attack)
    forced_actions: Dict[int, float] = field(default_factory=dict)

    # Optional metadata for designer tools / HUD
    archetype_name: str = "custom"
    sense_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)

    def apply_sensory(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sensory multipliers and biases to input tensor [..., d_in], clamped to >= 0."""
        out = x
        modified = False
        if self.sensory_scale is not None:
            scale = torch.as_tensor(self.sensory_scale, device=x.device, dtype=x.dtype)
            out = out * scale
            modified = True
        if self.sensory_bias is not None:
            bias = torch.as_tensor(self.sensory_bias, device=x.device, dtype=x.dtype)
            out = out + bias
            modified = True
        if modified:
            out = torch.clamp(out, min=0.0)
        return out

    def apply_actions(self, out: torch.Tensor) -> torch.Tensor:
        """Apply action scaling, clamping, and explicit forced overrides to output tensor [..., d_out]."""
        res = out
        if self.action_scale is not None:
            scale = torch.as_tensor(self.action_scale, device=out.device, dtype=out.dtype)
            res = res * scale
        if self.action_min is not None:
            amin = torch.as_tensor(self.action_min, device=out.device, dtype=out.dtype)
            res = torch.maximum(res, amin)
        if self.action_max is not None:
            amax = torch.as_tensor(self.action_max, device=out.device, dtype=out.dtype)
            res = torch.minimum(res, amax)
        if self.forced_actions:
            res = res.clone()
            for idx, val in self.forced_actions.items():
                if 0 <= idx < res.shape[-1]:
                    res[..., idx] = val
        return res

    def to_ini(self, path: Union[str, Path], model_cfg: Optional[Dict[str, int]] = None, chemistry: Optional[Dict[str, float]] = None) -> None:
        """Save designer tuning configuration to a studio-friendly .ini file."""
        cp = configparser.ConfigParser()
        cp["archetype"] = {
            "name": self.archetype_name,
            "senses": ",".join(self.sense_names) if self.sense_names else "",
            "actions": ",".join(self.action_names) if self.action_names else "",
        }
        if model_cfg:
            cp["architecture"] = {k: str(v) for k, v in model_cfg.items()}
        chem_section = {
            "apl_min": f"{self.apl_min:.4f}",
            "apl_max": f"{self.apl_max:.4f}",
            "apl_locked": "true" if self.apl_locked else "false",
            "chemical_bias": f"{self.chemical_bias:.4f}",
        }
        if chemistry:
            for k, v in chemistry.items():
                chem_section[k] = f"{v:.6f}"
        cp["chemistry"] = chem_section

        senses_sec = {}
        if self.sensory_scale is not None:
            senses_sec["sensory_scale"] = ",".join(f"{v:.4f}" for v in self.sensory_scale)
        if self.sensory_bias is not None:
            senses_sec["sensory_bias"] = ",".join(f"{v:.4f}" for v in self.sensory_bias)
        cp["senses"] = senses_sec

        actions_sec = {}
        if self.action_scale is not None:
            actions_sec["action_scale"] = ",".join(f"{v:.4f}" for v in self.action_scale)
        if self.action_min is not None:
            actions_sec["action_min"] = ",".join(f"{v:.4f}" for v in self.action_min)
        if self.action_max is not None:
            actions_sec["action_max"] = ",".join(f"{v:.4f}" for v in self.action_max)
        if self.forced_actions:
            actions_sec["forced_actions"] = ",".join(f"{k}:{v:.4f}" for k, v in sorted(self.forced_actions.items()))
        cp["actions"] = actions_sec

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# BioNPC Designer Tuning Profile: {self.archetype_name}\n")
            f.write("# Game designers can tune sensory multipliers, chemical decay, and action clamps here\n")
            f.write("# without retraining the underlying PyTorch CTRNN weights.\n\n")
            cp.write(f)

    @classmethod
    def from_ini(cls, path: Union[str, Path]) -> "DesignerControl":
        """Load designer tuning configuration from an .ini file."""
        cp = configparser.ConfigParser()
        cp.read(path, encoding="utf-8")

        def parse_float_list(s: str) -> Optional[List[float]]:
            s = s.strip()
            if not s:
                return None
            return [float(x.strip()) for x in s.split(",") if x.strip()]

        ctrl = cls()
        if "archetype" in cp:
            sec = cp["archetype"]
            ctrl.archetype_name = sec.get("name", "custom").strip()
            senses_str = sec.get("senses", "").strip()
            if senses_str:
                ctrl.sense_names = [x.strip() for x in senses_str.split(",")]
            actions_str = sec.get("actions", "").strip()
            if actions_str:
                ctrl.action_names = [x.strip() for x in actions_str.split(",")]

        if "chemistry" in cp:
            sec = cp["chemistry"]
            ctrl.apl_min = float(sec.get("apl_min", -10.0))
            ctrl.apl_max = float(sec.get("apl_max", 10.0))
            ctrl.apl_locked = sec.get("apl_locked", "false").strip().lower() in ("true", "1", "yes")
            ctrl.chemical_bias = float(sec.get("chemical_bias", 0.0))

        if "senses" in cp:
            sec = cp["senses"]
            if "sensory_scale" in sec:
                ctrl.sensory_scale = parse_float_list(sec["sensory_scale"])
            if "sensory_bias" in sec:
                ctrl.sensory_bias = parse_float_list(sec["sensory_bias"])

        if "actions" in cp:
            sec = cp["actions"]
            if "action_scale" in sec:
                ctrl.action_scale = parse_float_list(sec["action_scale"])
            if "action_min" in sec:
                ctrl.action_min = parse_float_list(sec["action_min"])
            if "action_max" in sec:
                ctrl.action_max = parse_float_list(sec["action_max"])
            if "forced_actions" in sec and sec["forced_actions"].strip():
                for pair in sec["forced_actions"].split(","):
                    pair = pair.strip()
                    if ":" in pair:
                        k_str, v_str = pair.split(":", 1)
                        ctrl.forced_actions[int(k_str.strip())] = float(v_str.strip())

        return ctrl
