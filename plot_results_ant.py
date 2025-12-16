import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ENV = "Ant-v5"
SEEDS = [0, 1, 2] 

def load_seed_curve(filepath):
    df = pd.read_csv(filepath)
    rewards = df["episode_reward"].values
    episodes = np.arange(len(rewards))
    return episodes, rewards

def aggregate_runs(filepaths):
    curves = []
    for fp in filepaths:
        episodes, rewards = load_seed_curve(fp)
        curves.append(rewards)

    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    episodes = np.arange(min_len)

    return episodes, mean, std

def plot_comparison(configs, title, outfile):
    plt.figure(figsize=(10, 6), dpi=300)

    for label, pattern in configs.items():
        files = []
        for s in SEEDS:
            fp = pattern.format(seed=s)
            if os.path.exists(fp):
                files.append(fp)

        if not files:
            continue

        episodes, mean, std = aggregate_runs(files)
        plt.plot(episodes, mean, label=f"{label} (n={len(files)})")
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Episodic Return", fontsize=12)
    plt.title(f"{ENV}: Training Curves (1,000,000 Steps)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# Buffer size comparison
buffer_configs = {
    "Buffer 500k": "data/baseline_rewards_Ant-v5_seed{seed}_buf500000_batch256.csv"
}

plot_comparison(
    buffer_configs,
    f"{ENV}: Buffer Size Comparison",
    "figures/ant_buffer_comparison_1M.png"
)

# Seed variation
plt.figure(figsize=(10, 6), dpi=300)

for s in SEEDS:
    fp = f"data/baseline_rewards_Ant-v5_seed{s}_buf500000_batch256.csv"
    if not os.path.exists(fp):
        continue

    df = pd.read_csv(fp)
    rewards = df["episode_reward"].values
    episodes = np.arange(len(rewards))

    plt.plot(episodes, rewards, label=f"Seed {s}")

plt.xlabel("Episode", fontsize=12)
plt.ylabel("Episodic Return", fontsize=12)
plt.title(f"{ENV}: Seed Variation (1,000,000 Steps)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/ant_seed_variation_1M.png", dpi=300, bbox_inches="tight")
plt.close()
