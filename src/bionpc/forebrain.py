"""
Hybrid LLM Forebrain + BioNPC Hindbrain Architecture.

The Forebrain (LLM or high-level narrative director) processes slow, semantic events
(dialogue, quest triggers, faction diplomacy) asynchronously and emits structured
`ForebrainSpike` packets.

The Hindbrain (BioNPC CTRNN) ingests those chemical spikes in O(1) time and runs
at 60+ FPS, handling continuous movement, combat reflexes, flanking, fleeing, and
natural neurotransmitter decay without any LLM token latency in the frame loop.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .control import AgentState, DesignerControl


@dataclass
class ForebrainSpike:
    """
    Structured chemical modulation command emitted by the LLM Forebrain
    to spike or clamp the 60 FPS BioNPC Hindbrain.
    """
    event_text: str
    narrative_thought: str
    sensory_bias_delta: Dict[str, float] = field(default_factory=dict)
    apl_spike: float = 0.0
    lock_chemical_state: bool = False
    forced_actions: Dict[str, float] = field(default_factory=dict)


class HybridForebrain:
    """
    Bridges high-level natural language events/dialogue (Forebrain) to continuous
    biological neurotransmitter states (Hindbrain).

    Supports:
    1. Live Ollama / OpenAI-compatible JSON endpoint (when `use_ollama=True`)
    2. Deterministic semantic rule-based neuro-modulator (zero-latency offline default)
    3. Custom LLM callable hook
    """
    def __init__(
        self,
        sense_names: List[str],
        action_names: List[str],
        use_ollama: bool = False,
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
        ollama_model: str = "phi3:mini",
        custom_llm_fn: Optional[Callable[[str], Dict]] = None,
    ):
        self.sense_names = list(sense_names)
        self.action_names = list(action_names)
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.custom_llm_fn = custom_llm_fn

    def evaluate_event(self, event_text: str) -> ForebrainSpike:
        """
        Translate a high-level dialogue line or world event into a `ForebrainSpike`.
        """
        if self.custom_llm_fn is not None:
            data = self.custom_llm_fn(event_text)
            return self._parse_dict(event_text, data)

        if self.use_ollama:
            try:
                return self._call_ollama(event_text)
            except Exception:
                pass

        return self._semantic_neuro_modulator(event_text)

    def _call_ollama(self, event_text: str) -> ForebrainSpike:
        prompt = (
            "You are the Forebrain of a biological game NPC.\n"
            f"Available sensory channels: {self.sense_names}\n"
            f"Available motor actions: {self.action_names}\n"
            f"Event: {event_text!r}\n"
            "Return JSON with keys: narrative_thought (str), sensory_bias_delta (dict of sense->float in [-1,1]), "
            "apl_spike (float in [-0.5, 0.5]), lock_chemical_state (bool), forced_actions (dict of action->float)."
        )
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body.get("response", "{}"))
            return self._parse_dict(event_text, parsed)

    def _parse_dict(self, event_text: str, data: Dict) -> ForebrainSpike:
        return ForebrainSpike(
            event_text=event_text,
            narrative_thought=str(data.get("narrative_thought", "Processed external event.")),
            sensory_bias_delta={
                k: float(v) for k, v in data.get("sensory_bias_delta", {}).items() if k in self.sense_names
            },
            apl_spike=float(data.get("apl_spike", 0.0)),
            lock_chemical_state=bool(data.get("lock_chemical_state", False)),
            forced_actions={
                k: float(v) for k, v in data.get("forced_actions", {}).items() if k in self.action_names
            },
        )

    def _semantic_neuro_modulator(self, event_text: str) -> ForebrainSpike:
        """
        Deterministic semantic analyzer mapping natural language threats, offerings,
        rally cries, and designer peace-zone commands into biological chemical spikes.
        """
        txt = event_text.lower()
        bias_delta: Dict[str, float] = {}
        forced: Dict[str, float] = {}
        apl_spike = 0.0
        lock_state = False
        thought = "Observing environment."

        # 1. Designer Scripted Override / Safe Zone / Cutscene Freeze (highest priority)
        if any(w in txt for w in ("safe zone", "sanctuary", "truce", "scripted stalk", "hold fire", "no attack")):
            thought = "Designer guardrail active: clamping lethal attack outputs and locking emotional state."
            lock_state = True
            for a in ("attack", "surge_attack", "ambush_weak"):
                if a in self.action_names:
                    forced[a] = -10.0  # Sigmoid(-10) ~ 0.000045 (strictly suppressed)

        # 2. Hostile / Enraging / Threatening events
        elif any(w in txt for w in ("burn", "kill", "attack", "slay", "strike", "blood", "cornered", "enrage", "berserk")):
            if any(w in txt for w in ("enrage", "berserk", "cornered", "alpha", "pack", "charge")):
                thought = "Adrenaline & octopamine surge! Fear suppressed; blind rage triggered."
                for s in ("aggression", "alertness", "queen_alert"):
                    if s in self.sense_names:
                        bias_delta[s] = 0.65
                for s in ("threat_sight", "intruder_sight", "enemy_sight", "intruder_proximity"):
                    if s in self.sense_names:
                        bias_delta[s] = 0.50
                apl_spike = -0.15  # Disinhibit motor circuits for maximum aggression
            else:
                thought = "High-threat stimulus detected! Cortisol/fear spike overriding foraging."
                for s in ("threat_sight", "predator_sight", "intruder_sight", "enemy_sight"):
                    if s in self.sense_names:
                        bias_delta[s] = 0.75
                for s in ("aggression", "morale"):
                    if s in self.sense_names:
                        bias_delta[s] = -0.35
                apl_spike = 0.25  # Heightened inhibitory vigilance

        # 3. Food / Appeasement / Bribe / Taming events
        elif any(w in txt for w in ("meat", "food", "feed", "peace", "calm", "offer", "treat", "graze")):
            thought = "Food/appeasement stimulus registered. Threat and aggression dampened; drawn to food."
            for s in ("aggression", "threat_sight", "predator_sight", "alertness"):
                if s in self.sense_names:
                    bias_delta[s] = -0.55
            for s in ("grass_sight", "carrion_sight", "prey_sight"):
                if s in self.sense_names:
                    bias_delta[s] = 0.35
            apl_spike = 0.10

        return ForebrainSpike(
            event_text=event_text,
            narrative_thought=thought,
            sensory_bias_delta=bias_delta,
            apl_spike=apl_spike,
            lock_chemical_state=lock_state,
            forced_actions=forced,
        )

    def apply_to_hindbrain(
        self,
        spike: ForebrainSpike,
        state: AgentState,
        control: DesignerControl,
        clear_previous_overrides: bool = True,
    ) -> None:
        """
        Apply a `ForebrainSpike` to the Hindbrain's `AgentState` and `DesignerControl`.
        """
        if control.sensory_bias is None or clear_previous_overrides:
            control.sensory_bias = [0.0] * len(self.sense_names)

        for sense_name, delta in spike.sensory_bias_delta.items():
            if sense_name in self.sense_names:
                idx = self.sense_names.index(sense_name)
                control.sensory_bias[idx] = max(-1.0, min(1.0, control.sensory_bias[idx] + delta))

        if spike.apl_spike != 0.0:
            state.spike(spike.apl_spike)

        control.apl_locked = spike.lock_chemical_state

        if clear_previous_overrides:
            control.forced_actions.clear()
        for act_name, val in spike.forced_actions.items():
            if act_name in self.action_names:
                idx = self.action_names.index(act_name)
                control.forced_actions[idx] = val

    def decay_sensory_biases(self, control: DesignerControl, decay_rate: float = 0.92) -> None:
        """
        Naturally decay Forebrain-injected sensory biases on each Hindbrain tick
        so temporary emotional spikes smoothly return to homeostasis.
        """
        if control.sensory_bias is not None:
            control.sensory_bias = [b * decay_rate for b in control.sensory_bias]
