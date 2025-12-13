import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

ENV = "Ant-v5"
SEEDS = [0, 1, 2]  # adjust based on available files
MAX_STEPS = 200_000

def load_seed_curve(filepath):
    df = pd.read_csv(filepath)
    if "step" in df.columns:
        steps = df["step"].values
    else:
        steps = np.arange(len(df))
    rewards = df.iloc[:, -1].values
    return steps, rewards

def aggregate_runs(filepaths):
    curves = []
    for fp in filepaths:
        steps, rewards = load_seed_curve(fp)
        mask = steps <= MAX_STEPS
        curves.append(rewards[mask])
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    steps = np.arange(min_len)
    return steps, mean, std

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
        steps, mean, std = aggregate_runs(files)
        plt.plot(steps, mean, label=f"{label} (n={len(files)})")
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

# Example: Buffer size comparison for Ant-v5
buffer_configs = {
    "Buffer 500k": "baseline_rewards_Ant-v5_seed{seed}_buf500000_batch256.csv"
}

plot_comparison(
    buffer_configs,
    f"{ENV}: Buffer Size Comparison",
    "ant_buffer_comparison.png"
)

# Example: Batch size comparison if needed
batch_configs = {
    "Batch 256": "baseline_rewards_Ant-v5_seed{seed}_buf500000_batch256.csv"
}

plot_comparison(
    batch_configs,
    f"{ENV}: Batch Size Comparison",
    "ant_batch_comparison.png"
)

# Seed variation
seed_files = [
    f"baseline_rewards_Ant-v5_seed{s}_buf500000_batch256.csv"
    for s in SEEDS
    if os.path.exists(f"baseline_rewards_Ant-v5_seed{s}_buf500000_batch256.csv")
]

if seed_files:
    plt.figure(figsize=(10, 6), dpi=300)
    for fp in seed_files:
        seed_num = int(fp.split("_seed")[1].split("_")[0])
        df = pd.read_csv(fp)
        steps = df["step"].values if "step" in df.columns else np.arange(len(df))
        rewards = df.iloc[:, -1].values
        mask = steps <= MAX_STEPS
        plt.plot(steps[mask], rewards[mask], label=f"Seed {seed_num}")
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title(f"{ENV}: Seed Variation (n={len(seed_files)})", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ant_seed_variation.png", dpi=300, bbox_inches="tight")
    plt.close()
