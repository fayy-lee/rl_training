import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use a built-in style (seaborn-whitegrid not available on my system)
plt.style.use('ggplot')

# Load baseline training CSVs
base_glob = "baseline_rewards_*_seed*_buf*_batch*.csv"
pattern = re.compile(
    r"baseline_rewards_(?P<env>.+?)_seed(?P<seed>\d+)_buf(?P<buf>\d+)_batch(?P<batch>\d+)\.csv"
)

files = sorted(glob.glob(base_glob))
if not files:
    print("No baseline_rewards CSVs found.")
    raise SystemExit(1)

# Parse into dict keyed by (env, buffer, batch)
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

    if "episode_reward" not in df.columns:
        raise ValueError(f"{f} missing 'episode_reward' column")

    rewards = df["episode_reward"].values
    key = (env, buf, batch)
    data.setdefault(key, {})[seed] = rewards

# Smoothing helper
def smooth(y, box=5):
    return pd.Series(y).rolling(box, min_periods=1).mean().values

out_dir = Path(".")

# 1) Hopper seed variation (buffer=500k, batch=256)
env_name = "Hopper-v4"
key = (env_name, 500000, 256)
if key in data:
    plt.figure(figsize=(10, 6))
    for seed, rewards in sorted(data[key].items()):
        plt.plot(smooth(rewards, box=10), label=f"seed {seed}")
    plt.title(f"{env_name} – Buffer 500k Batch 256 – Seed Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "hopper_seed_variation.png"
    plt.savefig(p)
    plt.close()
    print("Saved", p)

# 2) Buffer comparison for Hopper (batch=256)
def plot_buffer_compare(env, buffers, outname):
    plt.figure(figsize=(10, 6))
    for buf in buffers:
        key = (env, buf, 256)
        if key not in data:
            continue

        seeds = data[key]
        min_len = min(len(v) for v in seeds.values())
        stacked = np.array([v[:min_len] for v in seeds.values()])
        mean_curve = stacked.mean(axis=0)

        plt.plot(smooth(mean_curve, box=20), label=f"buffer {buf} (n={stacked.shape[0]})")

    plt.title(f"{env} – Buffer Comparison (batch=256)")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    print("Saved", outname)

plot_buffer_compare(
    "Hopper-v4",
    [300000, 500000],
    out_dir / "hopper_buffer_comparison.png"
)

# 3) Batch size comparison for Hopper (buffer=500k)
def plot_batch_compare(env, buffer, batches, outname):
    plt.figure(figsize=(10, 6))
    for batch in batches:
        key = (env, buffer, batch)
        if key not in data:
            continue

        seeds = data[key]
        min_len = min(len(v) for v in seeds.values())
        stacked = np.array([v[:min_len] for v in seeds.values()])
        mean_curve = stacked.mean(axis=0)

        plt.plot(
            smooth(mean_curve, box=20),
            label=f"batch {batch} (n={stacked.shape[0]})"
        )

    plt.title(f"{env} – Batch Size Comparison (buffer={buffer})")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    print("Saved", outname)

plot_batch_compare(
    "Hopper-v4",
    500000,
    [128, 256, 512],
    out_dir / "hopper_batch_comparison.png"
)

# 4) Evaluation results bar charts
eval_file = "evaluation_results.csv"
if Path(eval_file).exists():
    eval_df = pd.read_csv(eval_file)

    # Bar summary: mean ± std for each (env, buffer, batch)
    plt.figure(figsize=(10, 6))
    groups = eval_df.groupby(["env", "buffer", "batch"])
    for (env, buf, batch), grp in groups:
        mean = grp["mean_reward"].mean()
        std = grp["mean_reward"].std()
        label = f"{env} buf{buf} batch{batch}"
        plt.bar(label, mean, yerr=std, capsize=5)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Eval Reward")
    plt.title("Evaluation Results: Mean Reward ± Std")
    plt.tight_layout()
    outp = out_dir / "eval_bar_summary.png"
    plt.savefig(outp)
    plt.close()
    print("Saved", outp)

    # Ant vs Hopper summary
    grouped = eval_df.groupby("env")["mean_reward"].agg(["mean", "std", "count"])
    plt.figure(figsize=(8, 5))
    plt.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=5)
    plt.ylabel("Mean Eval Reward")
    plt.title("Eval Mean by Environment")
    plt.tight_layout()
    plt.savefig(out_dir / "ant_vs_hopper_eval.png")
    plt.close()
    print("Saved ant_vs_hopper_eval.png")

else:
    print("No evaluation_results.csv found — run evaluate_sweep.py first.")

print("All plots completed.")
