import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ENV = "Hopper-v4"
DATA_DIR = "results/data_t"
SEEDS = [0, 1, 2, 3, 4]

def load_seed_curve(filepath):
    if not os.path.exists(filepath):
        return None, None
    df = pd.read_csv(filepath)
    rewards = df["episode_reward"].values
    episodes = np.arange(len(rewards))
    return episodes, rewards

def aggregate_runs(filepaths):
    curves = []
    for fp in filepaths:
        episodes, rewards = load_seed_curve(fp)
        if rewards is not None:
            curves.append(rewards)
    
    if not curves:
        return None, None, None, 0
    
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    episodes = np.arange(min_len)

    return episodes, mean, std, len(curves)

# Plot 1: Batch size comparison
print("Creating batch size comparison plot...")
plt.figure(figsize=(10, 6), dpi=300)

BATCH_CONFIGS = {
    "Batch 128": "_bs128",
    "Batch 256": "",
    "Batch 512": "_bs512"
}

for batch_label, batch_suffix in BATCH_CONFIGS.items():
    files = []
    for seed in SEEDS:
        fp = f"{DATA_DIR}/sac_learning_curve_{ENV}_A_seed{seed}{batch_suffix}.csv"
        if os.path.exists(fp):
            files.append(fp)
    
    if not files:
        continue
    
    episodes, mean, std, n_seeds = aggregate_runs(files)
    if episodes is not None:
        print(f"  {batch_label}: {n_seeds} seeds")
        plt.plot(episodes, mean, label=f"{batch_label} (n={n_seeds})", linewidth=2)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episodic Return", fontsize=14)
plt.title(f"{ENV}: Learning Curves under Varying Batch Sizes", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/hopper_batch_size_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: results/figures/hopper_batch_size_comparison.png")
plt.close()

# Plot 2: Seed-wise variability for batch 256
print("\nCreating seed-wise variability plot for batch 256...")
plt.figure(figsize=(10, 6), dpi=300)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, seed in enumerate(SEEDS):
    fp = f"{DATA_DIR}/sac_learning_curve_{ENV}_A_seed{seed}.csv"
    episodes, rewards = load_seed_curve(fp)
    
    if rewards is not None:
        plt.plot(episodes, rewards, label=f"Seed {seed}", linewidth=1.5, 
                color=colors[i], alpha=0.8)

plt.xlabel("Episode", fontsize=14)
plt.ylabel("Episodic Return", fontsize=14)
plt.title(f"{ENV}: Seed-wise Variability with Batch 256", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/hopper_seed_variability_batch256.png", dpi=300, bbox_inches="tight")
print("Saved: results/figures/hopper_seed_variability_batch256.png")
plt.close()

print("\n✓ All plots generated!")

