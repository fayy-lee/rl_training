import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('ggplot')

# Load baseline CSVs
base_glob = "baseline_rewards_*_seed*_buf*_batch*.csv"
pattern = re.compile(
    r"baseline_rewards_(?P<env>.+?)_seed(?P<seed>\d+)_buf(?P<buf>\d+)_batch(?P<batch>\d+)\.csv"
)

files = sorted(glob.glob(base_glob))
if not files:
    print("No baseline_rewards CSVs found.")
    raise SystemExit(1)

data = {}
for f in files:
    m = pattern.search(Path(f).name)
    if not m:
        continue
    env = m.group("env")
    seed = int(m.group("seed"))
    buf = int(m.group("buf"))
    batch = int(m.group("batch"))
    df = pd.read_csv(f)
    rewards = df["episode_reward"].values
    key = (env, buf, batch)
    data.setdefault(key, {})[seed] = rewards

def smooth(y, box=5):
    return pd.Series(y).rolling(box, min_periods=1).mean().values

# Combined buffer comparison plot
plt.figure(figsize=(10,6))

# Hopper buffer 300k & 500k, batch 256
for buf in [300000, 500000]:
    key = ("Hopper-v4", buf, 256)
    if key in data:
        seeds = data[key]
        min_len = min(len(v) for v in seeds.values())
        stacked = np.array([v[:min_len] for v in seeds.values()])
        mean_curve = stacked.mean(axis=0)
        plt.plot(smooth(mean_curve, box=20), label=f"Hopper buf {buf} (n={stacked.shape[0]})")

# AntMaze buffer 500k, batch 256
key = ("Ant-v5", 500000, 256)
if key in data:
    seeds = data[key]
    min_len = min(len(v) for v in seeds.values())
    stacked = np.array([v[:min_len] for v in seeds.values()])
    mean_curve = stacked.mean(axis=0)
    plt.plot(smooth(mean_curve, box=20), label=f"AntMaze buf 500k (n={stacked.shape[0]})", linestyle="--")

plt.title("Buffer Comparison – Hopper vs AntMaze (batch 256)")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.legend()
plt.tight_layout()
plt.savefig("combined_buffer_comparison.png")
plt.close()
print("Saved combined_buffer_comparison.png")

# Combined batch comparison plot
plt.figure(figsize=(10,6))

# Hopper batch 128,256,512 buffer 500k
for batch in [128,256,512]:
    key = ("Hopper-v4", 500000, batch)
    if key in data:
        seeds = data[key]
        min_len = min(len(v) for v in seeds.values())
        stacked = np.array([v[:min_len] for v in seeds.values()])
        mean_curve = stacked.mean(axis=0)
        plt.plot(smooth(mean_curve, box=20), label=f"Hopper batch {batch} (n={stacked.shape[0]})")

# AntMaze batch 256, buffer 500k
key = ("Ant-v5", 500000, 256)
if key in data:
    seeds = data[key]
    min_len = min(len(v) for v in seeds.values())
    stacked = np.array([v[:min_len] for v in seeds.values()])
    mean_curve = stacked.mean(axis=0)
    plt.plot(smooth(mean_curve, box=20), label=f"AntMaze batch 256 (n={stacked.shape[0]})", linestyle="--")

plt.title("Batch Size Comparison – Hopper vs AntMaze (buffer 500k)")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.legend()
plt.tight_layout()
plt.savefig("combined_batch_comparison.png")
plt.close()
print("Saved combined_batch_comparison.png")
