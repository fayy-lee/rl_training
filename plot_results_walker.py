import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ENV = "Walker2d"
FILES = {
    "Variant A": r"C:\Users\fayel\Documents\rl_training\data\sac_learning_curve_Walker2d-v4_A_seed4.csv",
    "Variant B": r"C:\Users\fayel\Documents\rl_training\data\sac_learning_curve_Walker2d-v4_B_seed4.csv",
    "Variant C": r"C:\Users\fayel\Documents\rl_training\data\sac_learning_curve_Walker2d-v4_C_seed4.csv"
}

plt.figure(figsize=(10, 6), dpi=300)

for label, filepath in FILES.items():
    if not os.path.exists(filepath):
        continue
    df = pd.read_csv(filepath)
    rewards = df["episode_reward"].values
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, label=label)

plt.xlabel("Episode", fontsize=12)
plt.ylabel("Episodic Return", fontsize=12)
plt.title(f"{ENV}: Variant Comparison (Seed 4)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/walker_variant_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
