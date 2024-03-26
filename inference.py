import torch
import time

from model.tas import TemporalActionSegmentation

model = TemporalActionSegmentation(
    model_dim=64,
    learning_rate=0.0001,
    weight_decay=0.0005,
    scheduler_step=30,
    joint_count=2*21,
    joint_features=3,
    temporal_state_dim=128,
    spatial_state_dim=32,
    temporal_layers=8,
    spatial_layers=2
)

# 30 frames sample [B T J D]
sample = torch.zeros((1, 30, 42, 3))

# warmup
for _ in range(100):
    output = model.forward(sample)

times = []
for _ in range(100):
    beg = time.time()
    output = model.forward(sample)
    times.append(time.time() - beg)

res = sum(times) / len(times)
print(f'mean time (ms): {res*1000}')
