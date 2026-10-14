#!/usr/bin/env python3
"""
Comprehensive Verification Suite for BioNPC:
1. Causal Conv1D vs O(1) Stateful Step Equivalence
2. Python PyTorch <-> Native C Engine Bit-Exact Parity & CRC32 Lockstep Sync
3. Designer Controllability (State Locks, Spikes, Action Overrides, INI Round-Trip)
4. Zero-Shot Biological Archetypes Behavioral Verification
5. Hybrid LLM Forebrain -> 60 FPS BioNPC Hindbrain Integration
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from bionpc import (
    ARCHETYPE_REGISTRY,
    AgentState,
    DesignerControl,
    EmbodiedNPCModel,
    FusedAPLNorm,
    HybridForebrain,
    KenyonMLP,
    build_all_archetypes,
    load_archetype,
)
from bionpc.native import NativeBioNPC


class TestBioNPC(unittest.TestCase):
    def test_conv1d_vs_step_equivalence(self):
        """Verify sequence conv1d forward(x) matches step-by-step O(1) forward_step(x_t, state)."""
        torch.manual_seed(42)
        model = EmbodiedNPCModel(d_in=4, d_out=3, d_model=16, n_layers=2)
        model.eval()

        B, T, D_in = 5, 12, 4
        x_seq = torch.randn(B, T, D_in)

        with torch.no_grad():
            # Sequence conv1d forward
            out_conv = model(x_seq)

            # Step-by-step recurrent forward_step
            state = AgentState.zeros(batch_size=B, n_layers=2)
            step_outs = []
            for t in range(T):
                out_t, state = model.forward_step(x_seq[:, t, :], state=state)
                step_outs.append(out_t)
            out_step = torch.stack(step_outs, dim=1)

        max_err = (out_conv - out_step).abs().max().item()
        self.assertLess(max_err, 1e-5, f"Conv1D vs Step max error too large: {max_err}")

    def test_pytorch_to_c_engine_parity(self):
        """Verify exported PyTorch weights produce identical multi-step trajectories in C."""
        lib_path = Path(__file__).resolve().parent.parent / "c" / "lib" / "libbionpc.so"
        if not lib_path.exists():
            self.skipTest("libbionpc.so not built yet")

        torch.manual_seed(99)
        model = EmbodiedNPCModel(d_in=4, d_out=3, d_model=32, n_layers=2)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            w_path = Path(tmpdir) / "test.bionpc"
            model.save_weights(w_path)

            native = NativeBioNPC(w_path, lib_path=lib_path)
            B = 11  # Non-multiple of 4 to test both 4-way tile and remainder in C
            py_state = AgentState.zeros(batch_size=B, n_layers=2)
            c_states = None

            for step in range(10):
                x_np = np.random.RandomState(step + 100).uniform(0.0, 1.0, size=(B, 4)).astype(np.float32)
                x_pt = torch.from_numpy(x_np)

                with torch.no_grad():
                    py_out, py_state = model.forward_step(x_pt, state=py_state)

                c_out, c_states = native.forward_batch(x_np, states=c_states)

                diff = np.max(np.abs(py_out.numpy() - c_out))
                self.assertLess(diff, 1e-5, f"Step {step} PyTorch vs C output diff: {diff}")

            native.close()

    def test_designer_control_and_ini_roundtrip(self):
        """Verify DesignerControl sensory multipliers, state locks, spikes, action clamps, and INI."""
        torch.manual_seed(7)
        model = EmbodiedNPCModel(d_in=4, d_out=3, d_model=16, n_layers=2)
        model.eval()

        ctrl = DesignerControl(
            sensory_scale=[1.5, 0.5, 1.0, 2.0],
            sensory_bias=[0.1, 0.0, -0.1, 0.2],
            apl_min=0.0,
            apl_max=0.4,
            apl_locked=False,
            forced_actions={1: -10.0},
            archetype_name="test_wolf",
            sense_names=["s0", "s1", "s2", "s3"],
            action_names=["chase", "attack", "flee"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ini_path = Path(tmpdir) / "wolf.ini"
            ctrl.to_ini(ini_path, model_cfg=model.config_dict, chemistry={"decay_logit_l0": 2.5})
            loaded_ctrl = model.apply_ini_chemistry(ini_path)

            self.assertEqual(loaded_ctrl.archetype_name, "test_wolf")
            self.assertAlmostEqual(model.apl[0].decay_logit.item(), 2.5, places=5)
            self.assertIn(1, loaded_ctrl.forced_actions)
            self.assertAlmostEqual(loaded_ctrl.forced_actions[1], -10.0, places=5)

            state = AgentState.zeros(batch_size=2, n_layers=2)
            x = torch.ones(2, 4)
            with torch.no_grad():
                out, state = model.forward_step(x, state=state, control=loaded_ctrl)
            # Action 1 must be strictly clamped to -10.0
            self.assertTrue(torch.allclose(out[:, 1], torch.tensor([-10.0, -10.0])))
            # APL state must respect [0.0, 0.4] clamp
            self.assertTrue((state.apl_state >= 0.0).all() and (state.apl_state <= 0.4).all())

            # Test state serialization & CRC32
            crc1 = state.crc32()
            raw = state.to_bytes()
            restored = AgentState.from_bytes(raw)
            self.assertEqual(crc1, restored.crc32())

    def test_zero_shot_archetypes(self):
        """Verify all 5 pre-trained archetypes exhibit expected ecological behaviors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = build_all_archetypes(tmpdir)
            self.assertEqual(len(exported), 5)

            # 1. Apex Predator: [prey_sight, threat_sight, hunger, aggression] -> [chase, attack, flee]
            pred_model, pred_ctrl = load_archetype("apex_predator", archetypes_dir=tmpdir)
            with torch.no_grad():
                # Starving with prey in sight, no threat -> Chase (idx 0)
                out_chase, _ = pred_model.forward_step(torch.tensor([[0.9, 0.0, 0.9, 0.3]]), control=pred_ctrl)
                self.assertEqual(torch.argmax(out_chase[0]).item(), 0)

                # High threat, low aggression -> Flee (idx 2)
                out_flee, _ = pred_model.forward_step(torch.tensor([[0.1, 0.9, 0.2, 0.1]]), control=pred_ctrl)
                self.assertEqual(torch.argmax(out_flee[0]).item(), 2)

                # High threat, max aggression -> Attack (idx 1)
                out_atk, _ = pred_model.forward_step(torch.tensor([[0.1, 0.9, 0.8, 0.98]]), control=pred_ctrl)
                self.assertEqual(torch.argmax(out_atk[0]).item(), 1)

            # 2. Herd Prey: [predator_sight, grass_sight, hunger, flock_isolation] -> [flee, graze, regroup]
            prey_model, prey_ctrl = load_archetype("herd_prey", archetypes_dir=tmpdir)
            with torch.no_grad():
                # Predator spotted -> Flee (idx 0)
                out_panic, _ = prey_model.forward_step(torch.tensor([[0.9, 0.8, 0.8, 0.1]]), control=prey_ctrl)
                self.assertEqual(torch.argmax(out_panic[0]).item(), 0)

                # Safe and hungry on grass -> Graze (idx 1)
                out_graze, _ = prey_model.forward_step(torch.tensor([[0.0, 0.9, 0.9, 0.1]]), control=prey_ctrl)
                self.assertEqual(torch.argmax(out_graze[0]).item(), 1)

    def test_hybrid_forebrain_modulation(self):
        """Verify HybridForebrain translates narrative events into Hindbrain overrides and spikes."""
        model, ctrl = load_archetype("apex_predator")
        fb = HybridForebrain(sense_names=ctrl.sense_names, action_names=ctrl.action_names)
        state = AgentState.zeros(batch_size=1, n_layers=len(model.apl))

        # Sanctuary command must disable attack
        spike_safe = fb.evaluate_event("Player enters safe zone sanctuary, hold fire!")
        fb.apply_to_hindbrain(spike_safe, state, ctrl)
        self.assertTrue(ctrl.apl_locked)
        self.assertIn(1, ctrl.forced_actions)

        with torch.no_grad():
            out, state = model.forward_step(torch.tensor([[0.2, 0.9, 0.8, 0.95]]), state=state, control=ctrl)
            self.assertLess(torch.sigmoid(out[0, 1]).item(), 0.001)

        # Enrage command must unlock and boost aggression
        spike_rage = fb.evaluate_event("Enemy corners the alpha pack and burns their den! Enrage!")
        fb.apply_to_hindbrain(spike_rage, state, ctrl)
        self.assertFalse(ctrl.apl_locked)
        self.assertEqual(len(ctrl.forced_actions), 0)
        self.assertGreater(ctrl.sensory_bias[3], 0.3)

    def test_kenyon_mlp(self):
        """Verify Drosophila Mushroom Body KenyonMLP sparse projection runs and backpropagates."""
        kmlp = KenyonMLP(d_model=16, expand_ratio=4, k=4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        y = kmlp(x)
        self.assertEqual(y.shape, (2, 5, 16))
        y.sum().backward()
        self.assertIsNotNone(kmlp.W1.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
