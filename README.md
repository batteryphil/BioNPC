# BioNPC: Biological Foundation Models for Game AI

BioNPC is a Continuous Time Recurrent Neural Network (CTRNN) architecture derived directly from the synaptic wiring of the *Drosophila* (Fruit Fly) connectome, heavily optimized for parallel NPC execution in video games.

## Why BioNPC?

Traditional game NPCs use Finite State Machines (FSMs) or Behavior Trees. They feel robotic because their states are discrete. Using Large Language Models (LLMs) for NPCs is incredibly slow, lacks true continuity, and costs massive memory bandwidth due to KV caches.

**BioNPC** runs on continuous biological chemistry. 
- It uses `FusedAPLNorm`, an ultra-fast recurrent memory state that mimics biological neurotransmitter decay (like Fear, Hunger, and Aggression).
- Instead of words, it ingests continuous sensory floats (Sight, Health, Distance).
- Instead of text, it outputs continuous motor floats (Move X, Attack, Flee).
- It is incredibly lightweight. You can run **100,000 continuous intelligent NPCs simultaneously at 700+ FPS** on a single consumer GPU.

## Usage

```python
import torch
from bionpc import EmbodiedNPCModel

# Define an NPC Brain: 4 Sensory Inputs, 3 Motor Outputs
wolf_brain = EmbodiedNPCModel(d_in=4, d_out=3, d_model=16)

# Batched execution for 100,000 wolves simultaneously!
# Input: [Batch_Size, Time, Senses (e.g. Smell, Vision, Hunger, Aggression)]
senses = torch.rand(100000, 1, 4) 
actions = wolf_brain(senses) 
```

## Hybrid LLM-Forebrain Architecture

BioNPC handles 60 FPS fast-twitch reflexes, physics, combat, and emotional chemistry (The Hindbrain). For complex dialogue and quest planning, an LLM (The Forebrain) can occasionally spike the BioNPC's chemical inputs.

---

# Native Engine: Fil-C Cosmopolitan APE

Alongside the Python PyTorch reference implementation, BioNPC includes a **blazing fast, 100% memory-safe native C engine** compiled as a Cosmopolitan **Actually Portable Executable (APE)**.

### Why Fil-C Cosmopolitan?

1. **Run Anywhere (Truly Portable)**: A single binary (`.com` / `.com.com`) runs natively on **Linux, macOS (Intel & Apple Silicon), Windows (PE32+), FreeBSD, NetBSD, and OpenBSD** without Docker, Python, PyTorch, or CUDA dependencies.
2. **100% Memory Safe**: Compiled with [Fil-C](https://github.com/pizlonator/llvm-project-deluge), every pointer is a 128-bit capability fat pointer. Spatial and temporal memory corruptions (buffer overflows, use-after-free, double-free) are mathematically impossible.
3. **Blazing Fast CPU Inference**: Zero-dependency vectorized C inference with POSIX multi-threading, executing **over 350,000 continuous NPC decisions per second** directly on standard CPUs.

### Performance Benchmarks

Tested on standard x86_64 host (8 CPU threads):

| Runtime | Target Scale | Frame Time | Game FPS | Throughput | Memory Safety | Dependencies |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Python / PyTorch (CPU)** | 10,000 NPCs | ~700 ms | ~1.4 FPS | ~14,000 / sec | Python GIL / C extensions | Python, PyTorch (multi-GB) |
| **Fil-C Cosmo APE (1 Core)** | 10,000 NPCs | ~358 ms | 2.8 FPS | 27,900 / sec | 100% Fil-C Safe | **None (Single binary)** |
| **Fil-C Cosmo APE (8 Cores)** | 10,000 NPCs | **53.1 ms** | **18.8 FPS** | **188,225 / sec** | **100% Fil-C Safe** | **None (Single binary)** |
| **Fil-C Cosmo Ecosystem Sim** | 5,000 NPCs | **14.2 ms** | **70.2 TPS** | **351,045 / sec** | **100% Fil-C Safe** | **None (Single binary)** |
| **Native GCC (Unsafe)** | 10,000 NPCs | 21.0 ms | 47.5 FPS | 474,769 / sec | Unprotected C | Host OS libc |

### Quickstart (Native C Engine)

#### 1. Compile the Portable Executables
```bash
cd c
make
```
This generates:
- `c/bin/bionpc_bench.com.com` (High-performance multi-threaded benchmark)
- `c/bin/wolf_sheep_sim.com.com` (Large-scale continuous predator-prey ecosystem)
- `c/bin/test_io.com.com` (Serialization & numerical reproducibility test suite)
- `c/lib/libbionpc.a` (Static library ready to link into Unreal, Godot, or custom C/C++ game engines)

#### 2. Run the Benchmark
```bash
./c/bin/bionpc_bench.com.com 10000 8
```

#### 3. Run the Living Ecosystem Simulation
```bash
./c/bin/wolf_sheep_sim.com.com --wolves 1000 --sheep 4000 --steps 100 --threads 8
```

#### 4. Export Trained Weights from PyTorch
```bash
python3 scripts/export_weights.py --format bin --out wolf.bionpc
# Or export directly to a zero-overhead C header:
python3 scripts/export_weights.py --format header --out wolf_weights.h
```

### C Integration API

```c
#include "bionpc.h"

// 1. Create Wolf Brain: 4 inputs (senses), 3 outputs (actions), d_model=32, 2 layers
bionpc_config_t cfg = { .d_in = 4, .d_out = 3, .d_model = 32, .n_layers = 2 };
bionpc_model_t *brain = bionpc_model_create(&cfg);
bionpc_load_weights(brain, "wolf.bionpc");

// 2. Initialize per-agent recurrent biological chemistry state
bionpc_agent_state_t wolf_state = {0};

// 3. Game Tick (60 FPS): Ingest continuous senses, output continuous actions
float senses[4] = { sight_sheep, dist_sheep, hunger, aggression };
float actions[3]; // [Chase, Attack, Flee]
bionpc_forward_single(brain, senses, actions, &wolf_state);
```
