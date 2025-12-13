import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ENV = "Hopper-v4"
SEEDS = [0, 1, 2]  # consistent n = 3
MAX_STEPS = 200_000

files = [
    f"baseline_rewards_Hopper-v4_seed{s}_buf500000_batch256.csv"
    for s in SEEDS
    if os.path.exists(f"baseline_rewards_Hopper-v4_seed{s}_buf500000_batch256.csv")
]

plt.figure(figsize=(10, 6), dpi=300)

for s, fp in zip(SEEDS, files):
    df = pd.read_csv(fp)
    steps = df["step"].values if "step" in df.columns else np.arange(len(df))
    rewards = df.iloc[:, -1].values
    mask = steps <= MAX_STEPS
    plt.plot(steps[mask], rewards[mask], label=f"Seed {s}")

plt.xlabel("Environment Steps", fontsize=12)
plt.ylabel("Reward", fontsize=12)
plt.title(f"Hopper-v4: Seed Variation (n={len(files)})", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hopper_seed_variance.png", dpi=300, bbox_inches="tight")
plt.close()
