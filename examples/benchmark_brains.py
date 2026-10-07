import torch
import time
from wolf_sheep_sim import EmbodiedNPCModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create ONE brain
brain = EmbodiedNPCModel(d_in=4, d_out=3, d_model=32, n_layers=2).to(device)
brain.eval()

# Simulate 100,000 Wolves simultaneously!
num_wolves = 100000
# [Batch, Time=1, Features=4]
batched_inputs = torch.rand(num_wolves, 1, 4).to(device)

print(f"Benchmarking ONE brain processing {num_wolves:,} animals simultaneously on {device.upper()}...")

# Warmup
with torch.no_grad():
    for _ in range(10):
        _ = brain(batched_inputs)

# Benchmark
start = time.time()
with torch.no_grad():
    for _ in range(60): # 60 frames (1 second of game time)
        _ = brain(batched_inputs)
end = time.time()

total_time = end - start
fps = 60 / total_time
print(f"Processed 60 frames for {num_wolves:,} animals in {total_time:.4f} seconds!")
print(f"Equivalent Game FPS: {fps:.0f} FPS")
