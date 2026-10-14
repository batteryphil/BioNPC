import math
import struct
import configparser
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .control import AgentState, DesignerControl

MAGIC = b"BIONPC01"


class KenyonMLP(nn.Module):
    """
    Biological Mushroom Body replacement for a dense MLP.
    Expands into sparse Kenyon cells where each neuron connects to exactly `k` random
    projection inputs, mirroring the Drosophila olfactory/associative Mushroom Body.
    """
    def __init__(self, d_model: int, expand_ratio: int = 4, k: int = 4, seed: int = 42):
        super().__init__()
        self.d_model = d_model
        self.expand_ratio = expand_ratio
        self.k = min(k, d_model)
        self.d_hidden = d_model * expand_ratio

        self.W1 = nn.Parameter(torch.randn(self.d_hidden, d_model) * (1.0 / math.sqrt(self.k)))
        self.b1 = nn.Parameter(torch.zeros(self.d_hidden))
        self.W2 = nn.Parameter(torch.randn(d_model, self.d_hidden) * (1.0 / math.sqrt(self.d_hidden)))
        self.b2 = nn.Parameter(torch.zeros(d_model))

        gen = torch.Generator()
        gen.manual_seed(seed)
        mask = torch.zeros(self.d_hidden, d_model)
        for i in range(self.d_hidden):
            indices = torch.randperm(d_model, generator=gen)[: self.k]
            mask[i, indices] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(F.linear(x, self.W1 * self.mask, self.b1))
        return F.linear(h, self.W2, self.b2)


class FusedAPLNorm(nn.Module):
    """
    Drosophila Anterior Paired Lateral (APL) Recurrent Inhibitory Normalization.
    Maintains a continuous decaying neurotransmitter/inhibition state:
        state_t = sigmoid(decay_logit) * state_{t-1} + relu(gain) * mean(|x_t|)
        out_t   = x_t * (1 - state_t)

    Supports:
    - Parallel O(1)-depth causal 1D convolution over sequences [B, T, D] during training.
    - Stateful O(1)-memory single-step inference over [B, D] for 60+ FPS game loops,
      matching the native C `bionpc_forward_single` engine bit-for-bit.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.decay_logit = nn.Parameter(torch.tensor(2.0))
        self.gain = nn.Parameter(torch.tensor(0.1))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)

    @property
    def effective_gain(self) -> torch.Tensor:
        return F.relu(self.gain)

    def forward(self, x: torch.Tensor, init_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sequence forward pass over x of shape [B, T, D].
        Optionally accepts init_state of shape [B] for seamless chunked continuity.
        """
        B, T, D = x.shape
        decay = torch.sigmoid(self.decay_logit)
        gain = F.relu(self.gain)
        activity = x.abs().mean(dim=-1)  # [B, T]
        powers = torch.arange(T, device=x.device, dtype=x.dtype)
        h = (decay ** powers).flip(0).view(1, 1, T)
        activity_pad = F.pad(activity.unsqueeze(1), (T - 1, 0))
        state = F.conv1d(activity_pad, h).squeeze(1) * gain  # [B, T]

        if init_state is not None:
            # Contribution of initial recurrent state decays as decay^(t+1)
            decay_steps = decay ** (powers + 1.0)  # [T]
            state = state + init_state.unsqueeze(-1) * decay_steps.unsqueeze(0)

        return x * (1.0 - state.unsqueeze(-1))

    def forward_step(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        locked: Optional[torch.Tensor] = None,
        apl_min: float = -10.0,
        apl_max: float = 10.0,
        chemical_bias: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) single-tick recurrent update for real-time game loops.
        x: [B, D]
        prev_state: [B] or None
        Returns: (inhibited_x [B, D], new_state [B])
        """
        decay = torch.sigmoid(self.decay_logit)
        gain = F.relu(self.gain)
        activity = x.abs().mean(dim=-1)  # [B]

        if prev_state is not None:
            computed = prev_state * decay + activity * gain + chemical_bias
            if apl_min > -1e9 or apl_max < 1e9:
                computed = torch.clamp(computed, min=apl_min, max=apl_max)
            if locked is not None:
                new_state = torch.where(locked, prev_state, computed)
            else:
                new_state = computed
        else:
            new_state = activity * gain + chemical_bias
            if apl_min > -1e9 or apl_max < 1e9:
                new_state = torch.clamp(new_state, min=apl_min, max=apl_max)

        out = x * (1.0 - new_state.unsqueeze(-1))
        return out, new_state


class EmbodiedNPCModel(nn.Module):
    """
    Biological Foundation Model for NPCs.
    Uses Continuous Time Recurrent Neural Networks (CTRNN) inspired by the Drosophila Connectome.
    """
    def __init__(self, d_in: int, d_out: int, d_model: int = 16, n_layers: int = 1):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model
        self.n_layers = n_layers

        self.proj_in = nn.Linear(d_in, d_model)
        self.apl = nn.ModuleList([FusedAPLNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_layers)
        ])
        self.proj_out = nn.Linear(d_model, d_out)

    @property
    def config_dict(self):
        return {
            "d_in": self.proj_in.in_features,
            "d_out": self.proj_out.out_features,
            "d_model": self.proj_in.out_features,
            "n_layers": len(self.apl),
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[AgentState] = None,
        control: Optional[DesignerControl] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        - If x is 3D [B, T, d_in] and state is None: runs vectorized causal conv1d over T.
        - If state is provided (or x is 2D [B, d_in]): runs stateful O(1) recurrent tick(s)
          and updates `state` in-place.
        """
        if state is not None or x.dim() == 2:
            squeeze_time = False
            if x.dim() == 3 and x.size(1) == 1:
                out_2d, _ = self.forward_step(x.squeeze(1), state=state, control=control)
                return out_2d.unsqueeze(1)
            elif x.dim() == 2:
                out_2d, _ = self.forward_step(x, state=state, control=control)
                return out_2d
            else:
                # Multi-step stateful rollout over T
                B, T, _ = x.shape
                outputs = []
                for t in range(T):
                    step_out, state = self.forward_step(x[:, t, :], state=state, control=control)
                    outputs.append(step_out)
                return torch.stack(outputs, dim=1)

        if control is not None:
            x = control.apply_sensory(x)

        h = self.proj_in(x)
        for apl, mlp in zip(self.apl, self.mlp):
            h = h + mlp(apl(h))
        out = self.proj_out(h)

        if control is not None:
            out = control.apply_actions(out)
        return out

    def forward_step(
        self,
        x: torch.Tensor,
        state: Optional[AgentState] = None,
        control: Optional[DesignerControl] = None,
    ) -> Tuple[torch.Tensor, AgentState]:
        """
        Execute a single 60 FPS game tick with O(1) recurrent memory and optional DesignerControl.
        x: [B, d_in] or [d_in]
        Returns: (actions, updated_state)
        """
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)

        B = x.size(0)
        n_layers = len(self.apl)
        if state is None:
            state = AgentState.zeros(batch_size=B, n_layers=n_layers, device=x.device, dtype=x.dtype)

        if control is not None:
            x = control.apply_sensory(x)
            apl_min = control.apl_min
            apl_max = control.apl_max
            chem_bias = control.chemical_bias
            global_lock = control.apl_locked
        else:
            apl_min = -1e12
            apl_max = 1e12
            chem_bias = 0.0
            global_lock = False

        h = self.proj_in(x)
        for l_idx, (apl, mlp) in enumerate(zip(self.apl, self.mlp)):
            prev_s = state.apl_state[:, l_idx]
            if global_lock:
                locked = torch.ones(B, device=x.device, dtype=torch.bool)
            elif state.apl_locked is not None:
                locked = state.apl_locked[:, l_idx]
            else:
                locked = None

            apl_h, new_s = apl.forward_step(
                h,
                prev_state=prev_s,
                locked=locked,
                apl_min=apl_min,
                apl_max=apl_max,
                chemical_bias=chem_bias,
            )
            state.apl_state[:, l_idx] = new_s
            h = h + mlp(apl_h)

        out = self.proj_out(h)
        if control is not None:
            out = control.apply_actions(out)

        if unbatched:
            out = out.squeeze(0)
        return out, state

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Export model weights to native .bionpc binary format for C / game engines."""
        cfg = [
            self.proj_in.in_features,
            self.proj_out.out_features,
            self.proj_in.out_features,
            len(self.apl),
        ]
        with open(filepath, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<4i", *cfg))
            f.write(self.proj_in.weight.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(self.proj_in.bias.detach().cpu().numpy().astype("<f4").tobytes())
            for apl, mlp in zip(self.apl, self.mlp):
                f.write(np.array([apl.decay_logit.item()], dtype="<f4").tobytes())
                f.write(np.array([apl.gain.item()], dtype="<f4").tobytes())
                f.write(mlp[0].weight.detach().cpu().numpy().astype("<f4").tobytes())
                f.write(mlp[0].bias.detach().cpu().numpy().astype("<f4").tobytes())
                f.write(mlp[2].weight.detach().cpu().numpy().astype("<f4").tobytes())
                f.write(mlp[2].bias.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(self.proj_out.weight.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(self.proj_out.bias.detach().cpu().numpy().astype("<f4").tobytes())

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """Load model weights from a native .bionpc binary file."""
        data = Path(filepath).read_bytes()
        if len(data) < 24 or data[:8] != MAGIC:
            raise ValueError(f"Invalid .bionpc file header in {filepath}")
        d_in, d_out, d_model, n_layers = struct.unpack("<4i", data[8:24])
        if (
            d_in != self.proj_in.in_features
            or d_out != self.proj_out.out_features
            or d_model != self.proj_in.out_features
            or n_layers != len(self.apl)
        ):
            raise ValueError(
                f"Architecture mismatch: file has {(d_in, d_out, d_model, n_layers)}, "
                f"model has {(self.proj_in.in_features, self.proj_out.out_features, self.proj_in.out_features, len(self.apl))}"
            )
        offset = 24

        def read_floats(count: int) -> torch.Tensor:
            nonlocal offset
            arr = np.frombuffer(data[offset : offset + count * 4], dtype="<f4").copy()
            offset += count * 4
            return torch.from_numpy(arr)

        with torch.no_grad():
            self.proj_in.weight.copy_(read_floats(d_model * d_in).view(d_model, d_in))
            self.proj_in.bias.copy_(read_floats(d_model))
            for apl, mlp in zip(self.apl, self.mlp):
                apl.decay_logit.copy_(read_floats(1).squeeze(0))
                apl.gain.copy_(read_floats(1).squeeze(0))
                mlp[0].weight.copy_(read_floats(4 * d_model * d_model).view(4 * d_model, d_model))
                mlp[0].bias.copy_(read_floats(4 * d_model))
                mlp[2].weight.copy_(read_floats(d_model * 4 * d_model).view(d_model, 4 * d_model))
                mlp[2].bias.copy_(read_floats(d_model))
            self.proj_out.weight.copy_(read_floats(d_out * d_model).view(d_out, d_model))
            self.proj_out.bias.copy_(read_floats(d_out))

    @classmethod
    def from_weights(cls, filepath: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "EmbodiedNPCModel":
        """Instantiate an EmbodiedNPCModel directly from a .bionpc binary file."""
        data = Path(filepath).read_bytes()
        if len(data) < 24 or data[:8] != MAGIC:
            raise ValueError(f"Invalid .bionpc file: {filepath}")
        d_in, d_out, d_model, n_layers = struct.unpack("<4i", data[8:24])
        model = cls(d_in=d_in, d_out=d_out, d_model=d_model, n_layers=n_layers)
        model.load_weights(filepath)
        return model.to(device)

    def apply_ini_chemistry(self, path: Union[str, Path]) -> DesignerControl:
        """
        Load an .ini designer profile, updating layer decay_logit and gain parameters
        if specified in [chemistry], and returning the parsed DesignerControl.
        """
        cp = configparser.ConfigParser()
        cp.read(path, encoding="utf-8")
        if "chemistry" in cp:
            sec = cp["chemistry"]
            with torch.no_grad():
                for i, apl in enumerate(self.apl):
                    if f"decay_logit_l{i}" in sec:
                        apl.decay_logit.fill_(float(sec[f"decay_logit_l{i}"]))
                    elif "decay_logit" in sec:
                        apl.decay_logit.fill_(float(sec["decay_logit"]))
                    if f"gain_l{i}" in sec:
                        apl.gain.fill_(float(sec[f"gain_l{i}"]))
                    elif "gain" in sec:
                        apl.gain.fill_(float(sec["gain"]))
        return DesignerControl.from_ini(path)
