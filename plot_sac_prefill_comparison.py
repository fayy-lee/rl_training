import pandas as pd
import matplotlib.pyplot as plt

# Reward CSVs to compare
csv_files = {
    "SAC Baseline": "baseline_rewards.csv",
    "SAC + 50k prefill": "rewards_50k.csv",
    "SAC + 100k prefill": "rewards_100k.csv",
    "SAC + prior full": "prior_rewards.csv",
    # Uncomment if you have PPO rewards for the same env
    # "PPO": "ppo_rewards.csv"
}

plt.figure(figsize=(10, 6))

for label, path in csv_files.items():
    try:
        df = pd.read_csv(path)
        rewards = df["episode_reward"].rolling(10, min_periods=1).mean()
        plt.plot(rewards, label=label)
    except FileNotFoundError:
        print(f"Warning: {path} not found. Skipping {label}.")

plt.title("SAC Learning Curves with Different Prefill Levels")
plt.xlabel("Episode")
plt.ylabel("Episode Reward (smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sac_prefill_comparison.png")
plt.show()
