import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
ppo_csv = "cartpole_evaluation_dataset.csv"        # PPO evaluation data
rlpd_plot_img = "rlpd_cartpole_learning_curve.png" # RLPD image (for reference, not used directly)
rlpd_csv = "rlpd_cartpole_learning_curve.csv"      

# Load PPO dataset 
if not os.path.exists(ppo_csv):
    raise FileNotFoundError(f"{ppo_csv} not found in current folder.")
ppo_data = pd.read_csv(ppo_csv)

# Auto-detect reward column
reward_col = next((c for c in ppo_data.columns if 'reward' in c.lower() or 'return' in c.lower()), None)
if reward_col is None:
    raise ValueError("Could not cd a reward or return column in PPO CSV.")
ppo_rewards = ppo_data[reward_col]

# Smooth function
def smooth(y, box_pts=10):
    return pd.Series(y).rolling(box_pts, min_periods=1).mean()

# Plot PPO rewards 
plt.figure(figsize=(10, 6))
plt.plot(smooth(ppo_rewards), label="PPO (Online RL)", color="blue")

# Load RLPD data
if os.path.exists(rlpd_csv):
    rlpd_data = pd.read_csv(rlpd_csv)
    reward_col_rlpd = next((c for c in rlpd_data.columns if 'reward' in c.lower() or 'return' in c.lower()), None)
    if reward_col_rlpd:
        plt.plot(smooth(rlpd_data[reward_col_rlpd]), label="RLPD (Offline RL)", color="green")
    else:
        print("Could not find reward column in RLPD CSV; skipping RLPD curve.")
else:
    print("RLPD CSV not found; plotting only PPO curve.")

# Labels & styling 
plt.xlabel("Episode")
plt.ylabel("Episodic Return (Reward)")
plt.title("CartPole-v1: PPO vs RLPD Learning Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save & show 
output_path = "ppo_vs_rlpd_comparison.png"
plt.savefig(output_path)
plt.close()
print(f"Saved comparison plot as '{output_path}'")
