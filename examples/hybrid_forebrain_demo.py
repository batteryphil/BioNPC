#!/usr/bin/env python3
"""
Hybrid LLM Forebrain + 60 FPS BioNPC Hindbrain Showcase.

Demonstrates:
1. Zero-shot loading of pre-trained `apex_predator` archetype (.bionpc + .ini profile)
2. Designer Controllability: Safe-zone action overrides & locked chemical states
3. Hybrid LLM Forebrain: Translating natural-language events into chemical spikes
4. 60 FPS BioNPC Hindbrain: O(1) continuous sensorimotor reflexes with natural decay
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from bionpc import AgentState, HybridForebrain, load_archetype


def main():
    print("=" * 78)
    print("  BioNPC: Hybrid LLM Forebrain + 60 FPS Biological Hindbrain Showcase")
    print("=" * 78)

    model, ctrl = load_archetype("apex_predator")
    forebrain = HybridForebrain(sense_names=ctrl.sense_names, action_names=ctrl.action_names)
    state = AgentState.zeros(batch_size=1, n_layers=len(model.apl))

    # Base continuous senses: [prey_sight, threat_sight, hunger, aggression]
    senses = torch.tensor([[0.75, 0.10, 0.65, 0.35]], dtype=torch.float32)

    events = {
        1: ("60Hz Hindbrain: Prowling & tracking sheep", None),
        4: (
            "[DESIGNER GUARDRAIL] Wolf enters Village Sanctuary Zone",
            "Player enters safe zone sanctuary — hold fire, no attack allowed.",
        ),
        7: (
            "[LLM FOREBRAIN] Player yells: 'Corner the alpha and burn its den!'",
            "The enemy corners the alpha pack and threatens to burn their den! Enrage!",
        ),
        11: (
            "[LLM FOREBRAIN] Player tosses raw meat to appease the wolf",
            "Player offers raw meat and peace to feed the starving wolf.",
        ),
    }

    print(f"{'Tick':>4} | {'Context / Event':<48} | {'Chase':>6} | {'Attack':>6} | {'Flee':>6} | {'APL L0':>7}")
    print("-" * 90)

    with torch.no_grad():
        for tick in range(1, 15):
            label = "60Hz Hindbrain: Continuous reflex + decay"
            if tick in events:
                label, prompt = events[tick]
                if prompt is not None:
                    if tick == 4:
                        senses = torch.tensor([[0.20, 0.85, 0.70, 0.85]], dtype=torch.float32)
                    elif tick == 7:
                        senses = torch.tensor([[0.10, 0.85, 0.75, 0.55]], dtype=torch.float32)
                    elif tick == 11:
                        senses = torch.tensor([[0.60, 0.15, 0.60, 0.30]], dtype=torch.float32)

                    spike = forebrain.evaluate_event(prompt)
                    forebrain.apply_to_hindbrain(spike, state, ctrl)
            else:
                forebrain.decay_sensory_biases(ctrl, decay_rate=0.85)

            logits, state = model.forward_step(senses, state=state, control=ctrl)
            probs = torch.sigmoid(logits[0]) * 100.0
            chase, attack, flee = probs[0].item(), probs[1].item(), probs[2].item()
            apl_l0 = state.apl_state[0, 0].item()

            print(f"{tick:4d} | {label:<48} | {chase:5.1f}% | {attack:5.1f}% | {flee:5.1f}% | {apl_l0:+7.4f}")

    print("-" * 90)
    print(f"Deterministic Multiplayer State Checksum (CRC32): 0x{state.crc32():08X}")
    print("=" * 78)


if __name__ == "__main__":
    main()
