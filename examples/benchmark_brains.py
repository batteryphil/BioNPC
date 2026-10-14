import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from bionpc import AgentState, EmbodiedNPCModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create ONE brain
    brain = EmbodiedNPCModel(d_in=4, d_out=3, d_model=32, n_layers=2).to(device)
    brain.eval()

    # Simulate 100,000 Wolves simultaneously!
    num_wolves = 100000
    batched_inputs = torch.rand(num_wolves, 4, device=device)
    state = AgentState.zeros(batch_size=num_wolves, n_layers=2, device=device)

    print(f"Benchmarking ONE brain processing {num_wolves:,} animals simultaneously on {device.upper()}...")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _, state = brain.forward_step(batched_inputs, state=state)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark 60 frames (1 second of game time)
    start = time.time()
    with torch.no_grad():
        for _ in range(60):
            _, state = brain.forward_step(batched_inputs, state=state)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    fps = 60 / total_time
    print(f"Processed 60 frames for {num_wolves:,} animals in {total_time:.4f} seconds!")
    print(f"Equivalent Game FPS: {fps:.0f} FPS ({num_wolves * 60 / total_time:,.0f} decisions/sec)")


if __name__ == "__main__":
    main()
