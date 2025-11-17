import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Detect relevant SAC learning curve files
files = [f for f in os.listdir() if f.startswith("sac_learning_curve_Hopper-v4_")]

if not files:
    print("No SAC learning curve files found.")
    exit()

# Step 2: Define labels and colours 
labels = {
    "A": "Random Prefill (A)",
    "B": "50k PPO Prefill (B)",
    "C": "100k PPO Prefill (C)"
}

plt.figure(figsize=(8, 5))

# Step 3: Process each SAC variant 
for f in files:
    variant = f.split("_")[-2]  # ex., A, B, or C
    print(f"Processing: {f}")
    
    try:
        # Try reading with header first
        df = pd.read_csv(f)
        # If single column, rename
        if df.shape[1] == 1:
            df.columns = ["Reward"]
        elif "episode_reward" in df.columns:
            df.rename(columns={"episode_reward": "Reward"}, inplace=True)
        elif "Reward" not in df.columns:
            df.columns = ["Reward"]

        # Force numeric conversion (ignore header strings)
        df["Reward"] = pd.to_numeric(df["Reward"], errors="coerce")
        df = df.dropna(subset=["Reward"]).reset_index(drop=True)
        df["Episode"] = range(len(df))
        df["SmoothedReward"] = df["Reward"].rolling(window=10, min_periods=1).mean()

        plt.plot(df["Episode"], df["SmoothedReward"], label=labels.get(variant, f"Variant {variant}"))

    except Exception as e:
        print(f"Skipping {f} due to error: {e}")

# Step 4: Plot formatting
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("SAC Prefill Comparison vs PPO (Hopper-v4)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Step 5: Save and show plot 
plt.savefig("sac_vs_ppo_comparison_fixed.png", dpi=300)
plt.show()

print("\n Plot saved as sac_vs_ppo_comparison_fixed.png")
