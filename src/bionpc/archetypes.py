"""
Zero-Shot Pre-Trained Biological Archetypes for Game Studios.

Ships 5 ready-to-drop-in NPC archetypes so game designers can instantiate
believable creatures without writing PyTorch training code:
1. `apex_predator`      (Wolves, Bears, Stalkers)
2. `herd_prey`          (Sheep, Deer, Ambient Wildlife)
3. `territorial_guard`  (Town Guards, Sentries, Bosses)
4. `cowardly_scavenger` (Hyenas, Goblins, Opportunists)
5. `swarm_minion`       (Insectoids, Zerglings, Horde Zombies)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .control import DesignerControl
from .model import EmbodiedNPCModel


@dataclass
class ArchetypeSpec:
    name: str
    description: str
    d_in: int
    d_out: int
    d_model: int
    n_layers: int
    sense_names: List[str]
    action_names: List[str]
    decay_logit: float
    gain: float
    rule_fn: Callable[[torch.Tensor], torch.Tensor]


def _rule_apex_predator(X: torch.Tensor) -> torch.Tensor:
    # X: [N, 4] -> [prey_sight, threat_sight, hunger, aggression]
    # y: [N, 3] -> [chase, attack, flee]
    prey, threat, hunger, aggro = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = torch.zeros(X.size(0), 3, device=X.device, dtype=X.dtype)
    flee_mask = (threat > 0.4) & (aggro < 0.75)
    attack_mask = (threat > 0.4) & (aggro >= 0.75)
    chase_mask = (~flee_mask) & (~attack_mask) & (prey > 0.25) & (hunger > 0.3)
    y[flee_mask, 2] = 1.0
    y[attack_mask, 1] = 1.0
    y[chase_mask, 0] = 1.0
    return y


def _rule_herd_prey(X: torch.Tensor) -> torch.Tensor:
    # X: [N, 4] -> [predator_sight, grass_sight, hunger, flock_isolation]
    # y: [N, 3] -> [flee, graze, regroup]
    pred, grass, hunger, isolation = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = torch.zeros(X.size(0), 3, device=X.device, dtype=X.dtype)
    flee_mask = pred > 0.2
    regroup_mask = (~flee_mask) & (isolation > 0.6)
    graze_mask = (~flee_mask) & (~regroup_mask) & (grass > 0.2) & (hunger > 0.2)
    y[flee_mask, 0] = 1.0
    y[graze_mask, 1] = 1.0
    y[regroup_mask, 2] = 1.0
    return y


def _rule_territorial_guard(X: torch.Tensor) -> torch.Tensor:
    # X: [N, 4] -> [intruder_sight, intruder_proximity, alertness, health]
    # y: [N, 4] -> [patrol, warn, attack, retreat]
    sight, prox, alert, health = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = torch.zeros(X.size(0), 4, device=X.device, dtype=X.dtype)
    retreat_mask = (health < 0.25) & (sight > 0.3)
    attack_mask = (~retreat_mask) & (sight > 0.3) & ((prox > 0.55) | (alert > 0.8))
    warn_mask = (~retreat_mask) & (~attack_mask) & (sight > 0.25)
    patrol_mask = (~retreat_mask) & (~attack_mask) & (~warn_mask)
    y[patrol_mask, 0] = 1.0
    y[warn_mask, 1] = 1.0
    y[attack_mask, 2] = 1.0
    y[retreat_mask, 3] = 1.0
    return y


def _rule_cowardly_scavenger(X: torch.Tensor) -> torch.Tensor:
    # X: [N, 4] -> [carrion_sight, target_weakness, threat_sight, hunger]
    # y: [N, 3] -> [scavenge, ambush_weak, flee]
    carrion, weakness, threat, hunger = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = torch.zeros(X.size(0), 3, device=X.device, dtype=X.dtype)
    flee_mask = threat > 0.25
    ambush_mask = (~flee_mask) & (weakness > 0.65) & (hunger > 0.35)
    scav_mask = (~flee_mask) & (~ambush_mask) & (carrion > 0.2)
    y[scav_mask, 0] = 1.0
    y[ambush_mask, 1] = 1.0
    y[flee_mask, 2] = 1.0
    return y


def _rule_swarm_minion(X: torch.Tensor) -> torch.Tensor:
    # X: [N, 4] -> [pheromone_signal, enemy_sight, swarm_density, queen_alert]
    # y: [N, 3] -> [surge_attack, flank_orbit, disperse]
    pheromone, enemy, density, queen = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    y = torch.zeros(X.size(0), 3, device=X.device, dtype=X.dtype)
    surge_mask = (enemy > 0.2) & ((density > 0.45) | (queen > 0.5))
    flank_mask = (~surge_mask) & (enemy > 0.2) & (pheromone > 0.25)
    disperse_mask = (~surge_mask) & (~flank_mask) & (enemy > 0.4) & (density < 0.3)
    y[surge_mask, 0] = 1.0
    y[flank_mask, 1] = 1.0
    y[disperse_mask, 2] = 1.0
    return y


ARCHETYPE_REGISTRY: Dict[str, ArchetypeSpec] = {
    "apex_predator": ArchetypeSpec(
        name="apex_predator",
        description="Stalks and chases prey when hungry; flees superior threats unless enraged.",
        d_in=4,
        d_out=3,
        d_model=32,
        n_layers=2,
        sense_names=["prey_sight", "threat_sight", "hunger", "aggression"],
        action_names=["chase", "attack", "flee"],
        decay_logit=2.2,  # Slow adrenaline decay (~0.90 per tick)
        gain=0.12,
        rule_fn=_rule_apex_predator,
    ),
    "herd_prey": ArchetypeSpec(
        name="herd_prey",
        description="Grazes peacefully and regroups with herd; triggers immediate high-gain panic flee when predator is spotted.",
        d_in=4,
        d_out=3,
        d_model=16,
        n_layers=2,
        sense_names=["predator_sight", "grass_sight", "hunger", "flock_isolation"],
        action_names=["flee", "graze", "regroup"],
        decay_logit=1.8,  # Moderate fear persistence (~0.86 per tick)
        gain=0.18,
        rule_fn=_rule_herd_prey,
    ),
    "territorial_guard": ArchetypeSpec(
        name="territorial_guard",
        description="Patrols perimeter, warns distant intruders, engages close threats, and retreats when critically wounded.",
        d_in=4,
        d_out=4,
        d_model=32,
        n_layers=2,
        sense_names=["intruder_sight", "intruder_proximity", "alertness", "health"],
        action_names=["patrol", "warn", "attack", "retreat"],
        decay_logit=2.6,  # High vigilance memory (~0.93 per tick)
        gain=0.08,
        rule_fn=_rule_territorial_guard,
    ),
    "cowardly_scavenger": ArchetypeSpec(
        name="cowardly_scavenger",
        description="Avoids fair fights, scavenges carrion, ambushes weakened targets, and flees at the first sign of danger.",
        d_in=4,
        d_out=3,
        d_model=16,
        n_layers=2,
        sense_names=["carrion_sight", "target_weakness", "threat_sight", "hunger"],
        action_names=["scavenge", "ambush_weak", "flee"],
        decay_logit=1.4,  # Nervous, twitchy decay (~0.80 per tick)
        gain=0.22,
        rule_fn=_rule_cowardly_scavenger,
    ),
    "swarm_minion": ArchetypeSpec(
        name="swarm_minion",
        description="Fast-twitch hive-mind creature driven by pheromones, local swarm density, and queen alert signals.",
        d_in=4,
        d_out=3,
        d_model=16,
        n_layers=1,
        sense_names=["pheromone_signal", "enemy_sight", "swarm_density", "queen_alert"],
        action_names=["surge_attack", "flank_orbit", "disperse"],
        decay_logit=0.8,  # Fast-twitch reflex decay (~0.69 per tick)
        gain=0.15,
        rule_fn=_rule_swarm_minion,
    ),
}


def train_archetype(
    name: str,
    steps: int = 120,
    seq_len: int = 2048,
    seed: int = 1337,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[EmbodiedNPCModel, DesignerControl]:
    """
    Deterministically train a biological archetype brain from its behavioral specification
    and return `(model, designer_control)`.
    """
    if name not in ARCHETYPE_REGISTRY:
        raise KeyError(f"Unknown archetype {name!r}. Available: {list(ARCHETYPE_REGISTRY.keys())}")

    spec = ARCHETYPE_REGISTRY[name]
    torch.manual_seed(seed)

    model = EmbodiedNPCModel(
        d_in=spec.d_in,
        d_out=spec.d_out,
        d_model=spec.d_model,
        n_layers=spec.n_layers,
    ).to(device)

    with torch.no_grad():
        for apl in model.apl:
            apl.decay_logit.fill_(spec.decay_logit)
            apl.gain.fill_(spec.gain)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1)
    # Sample across [-0.2, 1.5] so Forebrain sensory biases generalize cleanly
    X_cpu = torch.rand(256, 16, spec.d_in, generator=gen) * 1.7 - 0.2
    X = X_cpu.to(device)
    y = spec.rule_fn(X.reshape(-1, spec.d_in)).reshape(256, 16, spec.d_out)

    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(steps):
        # Train jointly on multi-step recurrent rollout (T=16) and single-step (T=1)
        logits_seq = model(X)
        logits_step = model(X.reshape(-1, 1, spec.d_in))
        loss = 0.5 * criterion(logits_seq, y) + 0.5 * criterion(logits_step, y.reshape(-1, 1, spec.d_out))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for apl in model.apl:
            apl.decay_logit.fill_(spec.decay_logit)
            apl.gain.fill_(spec.gain)

    model.eval()
    ctrl = DesignerControl(
        sensory_scale=[1.0] * spec.d_in,
        sensory_bias=[0.0] * spec.d_in,
        apl_min=-5.0,
        apl_max=5.0,
        apl_locked=False,
        chemical_bias=0.0,
        action_scale=[1.0] * spec.d_out,
        action_min=[-20.0] * spec.d_out,
        action_max=[20.0] * spec.d_out,
        archetype_name=spec.name,
        sense_names=list(spec.sense_names),
        action_names=list(spec.action_names),
    )
    return model, ctrl


def build_all_archetypes(output_dir: Union[str, Path]) -> Dict[str, Tuple[Path, Path]]:
    """
    Pre-train and export all 5 canonical archetypes to `.bionpc` binary weights
    and `.ini` designer profiles in `output_dir`.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Tuple[Path, Path]] = {}

    for idx, (name, spec) in enumerate(ARCHETYPE_REGISTRY.items()):
        model, ctrl = train_archetype(name, steps=120, seed=1000 + idx * 100)
        bin_file = out_path / f"{name}.bionpc"
        ini_file = out_path / f"{name}.ini"

        model.save_weights(bin_file)
        chemistry = {
            f"decay_logit_l{i}": model.apl[i].decay_logit.item()
            for i in range(spec.n_layers)
        }
        chemistry.update({
            f"gain_l{i}": model.apl[i].gain.item()
            for i in range(spec.n_layers)
        })
        ctrl.to_ini(ini_file, model_cfg=model.config_dict, chemistry=chemistry)
        exported[name] = (bin_file, ini_file)

    return exported


def load_archetype(
    name: str,
    archetypes_dir: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[EmbodiedNPCModel, DesignerControl]:
    """
    Load a pre-trained biological archetype and its DesignerControl profile.
    If pre-exported files exist in `archetypes_dir`, loads them in < 1ms without training.
    """
    if archetypes_dir is None:
        archetypes_dir = Path(__file__).resolve().parent.parent.parent / "archetypes"
    else:
        archetypes_dir = Path(archetypes_dir)

    bin_file = archetypes_dir / f"{name}.bionpc"
    ini_file = archetypes_dir / f"{name}.ini"

    if bin_file.exists() and ini_file.exists():
        model = EmbodiedNPCModel.from_weights(bin_file, device=device)
        ctrl = model.apply_ini_chemistry(ini_file)
        model.eval()
        return model, ctrl

    return train_archetype(name, device=device)
