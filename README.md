# BioNPC: Biological Foundation Models for Game AI

BioNPC is a Continuous Time Recurrent Neural Network (CTRNN) architecture derived directly from the synaptic wiring of the *Drosophila* (Fruit Fly) connectome, heavily optimized for parallel NPC execution in video games.

## Why BioNPC?

Traditional game NPCs use Finite State Machines (FSMs) or Behavior Trees. They feel robotic because their states are discrete. Using Large Language Models (LLMs) for NPCs is incredibly slow, lacks true continuity, and costs massive memory bandwidth due to KV caches.

**BioNPC** runs on continuous biological chemistry:
- **`FusedAPLNorm` & `KenyonMLP`**: Ultra-fast recurrent memory states inspired by the *Drosophila* Anterior Paired Lateral (APL) neuron and Mushroom Body that mimic biological neurotransmitter decay (Fear, Hunger, Aggression).
- **Continuous Sensorimotor Loop**: Ingests continuous sensory floats (`Sight`, `Proximity`, `Hunger`, `Aggression`) and outputs continuous motor floats (`Chase`, `Attack`, `Flee`) in O(1) memory per tick.
- **Extreme Throughput**: Run **100,000+ continuous intelligent NPCs simultaneously at 700+ FPS** on a single consumer GPU, or **450,000+ NPC decisions/sec** on standard CPU cores via the zero-dependency Fil-C Cosmopolitan APE native engine.

---

## Game-Studio Readiness: The 4 Pillars

BioNPC is engineered specifically to solve the four core adoption hurdles required by commercial game studios:

### 1. Designer Controllability & Guardrails (`DesignerControl` / `bionpc_control_t`)
Game designers never lose control to a "black box":
- **Action Clamping & Safe-Zone Overrides**: Force or forbid specific actions on demand (e.g., `forced_actions = {ATTACK: -10.0}` in village sanctuary zones).
- **Chemical State Locking**: Freeze neurotransmitter decay (`apl_locked = True`) during cutscenes or scripted boss phases.
- **Sensory Multipliers & Biases**: Tune creature sensitivity via human-readable `.ini` files without retraining PyTorch weights.

### 2. Zero-Shot Pre-Trained Biological Archetypes (`archetypes/`)
No PyTorch or Reinforcement Learning required for game designers. BioNPC ships with 5 pre-trained, ready-to-drop-in `.bionpc` binary brains and `.ini` tuning profiles:
| Archetype | Sensory Inputs (`d_in=4`) | Motor Outputs | Temperament / Chemistry |
| :--- | :--- | :--- | :--- |
| **`apex_predator`** | `prey_sight, threat_sight, hunger, aggression` | `chase, attack, flee` | Slow adrenaline decay (`decay_logit=2.2`); stalks prey, attacks when enraged |
| **`herd_prey`** | `predator_sight, grass_sight, hunger, flock_isolation` | `flee, graze, regroup` | High-gain panic reflex (`gain=0.18`); grazes when safe, regroups when isolated |
| **`territorial_guard`** | `intruder_sight, intruder_proximity, alertness, health` | `patrol, warn, attack, retreat` | High vigilance memory (`decay_logit=2.6`); warns before engaging, retreats if critical |
| **`cowardly_scavenger`** | `carrion_sight, target_weakness, threat_sight, hunger` | `scavenge, ambush_weak, flee` | Twitchy decay (`decay_logit=1.4`); ambushes weakened targets, flees healthy threats |
| **`swarm_minion`** | `pheromone_signal, enemy_sight, swarm_density, queen_alert` | `surge_attack, flank_orbit, disperse` | Fast-twitch hive reflex (`decay_logit=0.8`); surges in dense swarms or queen alerts |

### 3. Hybrid LLM-Forebrain + 60 FPS BioNPC-Hindbrain (`HybridForebrain`)
- **The Hindbrain (BioNPC, 60+ FPS)**: Handles fast-twitch reflexes, movement, combat dodging, and continuous emotional chemistry in sub-millisecond time.
- **The Forebrain (LLM / Narrative Director, ~1 Hz or on-event)**: Processes complex player dialogue or world events (e.g., *"Burn their den!"* or *"Player tosses raw meat"*) and injects a single structured `ForebrainSpike` into the BioNPC's chemical state—zero LLM latency in the frame loop.

```bash
python3 examples/hybrid_forebrain_demo.py
# Or run the native Fil-C Cosmopolitan APE demo:
./c/bin/hybrid_demo.com
```

### 4. Deterministic Multiplayer & Drop-In Engine Integrations (`integrations/`)
- **Cross-Platform Determinism**: Uses a portable SplitMix32 PRNG and IEEE-754 CRC32 state verification (`bionpc_crc32_states` / `state.crc32()`) for lockstep multiplayer servers and rollback netcode.
- **Binary State Snapshots**: Save and restore thousands of NPC recurrent states in microseconds (`bionpc_save_states` / `bionpc_load_states`).
- **Unreal Engine 5 & Godot 4 Integrations**:
  - [`integrations/unreal/BioNPCComponent.h`](integrations/unreal/BioNPCComponent.h): Ready-to-use UE5 `UActorComponent` with Blueprint-callable `StepBrain`, `InjectChemicalSpike`, and `SetActionOverride`.
  - [`integrations/godot/BioNPCAgent.gd`](integrations/godot/BioNPCAgent.gd): Godot 4 `CharacterBody3D` controller with a live 3D overhead debug HUD rendering `FEAR`, `HUNGER`, `AGGRO`, and `APL` bars.

---

## Python Quickstart

```bash
pip install -e .
```

```python
import torch
from bionpc import load_archetype, AgentState, HybridForebrain

# 1. Load a pre-trained biological archetype + designer .ini profile
wolf_brain, ctrl = load_archetype("apex_predator")
state = AgentState.zeros(batch_size=100000, n_layers=2)

# 2. Execute a single O(1) 60 FPS game tick for 100,000 wolves simultaneously
# Senses: [prey_sight, threat_sight, hunger, aggression]
senses = torch.rand(100000, 4)
actions, state = wolf_brain.forward_step(senses, state=state, control=ctrl)

# 3. Verify deterministic multiplayer lockstep checksum
print(f"Lockstep State CRC32: 0x{state.crc32():08X}")
```

---

## Native Engine: Fil-C Cosmopolitan APE & C/C++ SDK

Alongside the Python PyTorch implementation, BioNPC includes a **blazing fast, 100% memory-safe native C engine** compiled both as a Cosmopolitan **Actually Portable Executable (APE)** and as standard static/shared libraries (`libbionpc.a` / `libbionpc.so`).

### Why Fil-C Cosmopolitan?

1. **Run Anywhere (Truly Portable)**: A single binary (`.com`) runs natively on **Linux, macOS (Intel & Apple Silicon), Windows (PE32+), FreeBSD, NetBSD, and OpenBSD** without Docker, Python, PyTorch, or CUDA dependencies.
2. **100% Memory Safe**: Compiled with [Fil-C](https://github.com/pizlonator/llvm-project-deluge), every pointer is a 128-bit capability fat pointer. Spatial and temporal memory corruptions (buffer overflows, use-after-free, double-free) are mathematically impossible.
3. **4-Way Tiled CPU Inference**: Zero-dependency vectorized 4-way tiled C inference with POSIX multi-threading, executing **over 450,000 continuous NPC decisions per second** directly on standard CPUs.

### Performance Benchmarks

Tested on standard x86_64 host (8 CPU threads):

| Runtime | Target Scale | Frame Time | Game FPS | Throughput | Memory Safety | Dependencies |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Python / PyTorch (CPU)** | 10,000 NPCs | ~700 ms | ~1.4 FPS | ~14,000 / sec | Python GIL / C extensions | Python, PyTorch (multi-GB) |
| **Fil-C Cosmo APE (1 Core)** | 10,000 NPCs | 265.7 ms | 3.8 FPS | 37,642 / sec | 100% Fil-C Safe | **None (Single binary)** |
| **Fil-C Cosmo APE (8 Cores)** | 10,000 NPCs | **48.7 ms** | **20.5 FPS** | **205,131 / sec** | **100% Fil-C Safe** | **None (Single binary)** |
| **Fil-C Cosmo Ecosystem Sim** | 5,000 NPCs | **11.1 ms** | **90.3 TPS** | **451,578 / sec** | **100% Fil-C Safe** | **None (Single binary)** |
| **Native GCC (Unsafe)** | 10,000 NPCs | 21.0 ms | 47.5 FPS | 474,769 / sec | Unprotected C | Host OS libc |

### Build & Run (Native C Engine)

```bash
cd c
make all test bench sim hybrid
```
This builds:
- `c/bin/bionpc_bench.com` (High-performance 4-way tiled multi-threaded benchmark)
- `c/bin/wolf_sheep_sim.com` (Large-scale continuous predator-prey ecosystem)
- `c/bin/hybrid_demo.com` (Hybrid LLM Forebrain + 60Hz Hindbrain & Designer Control showcase)
- `c/bin/test_io.com` (Cross-platform determinism, CRC32, INI profile, and serialization test suite)
- `c/lib/libbionpc.a` & `c/lib/libbionpc.so` (Static & shared libraries ready to link into Unreal, Godot, Unity, or C/C++ engines)

### C Integration API

```c
#include "bionpc.h"

// 1. Load pre-trained archetype weights & Designer .ini tuning profile
bionpc_model_t *brain = bionpc_model_load_file("archetypes/apex_predator.bionpc");
bionpc_control_t ctrl;
bionpc_load_ini_profile("archetypes/apex_predator.ini", brain, &ctrl);

// 2. Initialize per-agent recurrent biological chemistry state
bionpc_agent_state_t wolf_state = {0};

// 3. Optional: Forebrain chemical spike or Designer safe-zone override
bionpc_state_spike(&wolf_state, 0, -0.15f); // Adrenaline surge
ctrl.action_override_mask[1] = 0;           // Allow attack

// 4. Game Tick (60 FPS): Ingest continuous senses, output continuous actions
float senses[4] = { sight_sheep, sight_threat, hunger, aggression };
float actions[3]; // [Chase, Attack, Flee]
bionpc_forward_single_controlled(brain, senses, actions, &wolf_state, &ctrl);

// 5. Lockstep multiplayer state verification
uint32_t sync_crc = bionpc_crc32_states(&wolf_state, 1, brain->config.n_layers);
```
