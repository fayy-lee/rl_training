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

# Combined seed variance plot
plt.figure(figsize=(10,6))

# Hopper seed variance (buffer 500k, batch 256)
key = ("Hopper-v4", 500000, 256)
if key in data:
    seeds = data[key]
    for seed, rewards in sorted(seeds.items()):
        plt.plot(smooth(rewards, box=10), alpha=0.7, label=f"Hopper seed {seed}")

# AntMaze seed variance (buffer 500k, batch 256)
key = ("Ant-v5", 500000, 256)
if key in data:
    seeds = data[key]
    for seed, rewards in sorted(seeds.items()):
        plt.plot(smooth(rewards, box=10), alpha=0.7, linestyle="--", label=f"AntMaze seed {seed}")

plt.title("Seed Variation – Hopper vs AntMaze (buffer 500k, batch 256)")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.legend()
plt.tight_layout()
plt.savefig("combined_seed_variance.png")
plt.close()
print("Saved combined_seed_variance.png")
