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
