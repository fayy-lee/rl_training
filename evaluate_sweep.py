import glob
import re
import argparse
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
from sac_utils import SACAgent  # sac implementation

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=10, help="Episodes per checkpoint")
parser.add_argument("--render", action="store_true", help="Render env during eval (shows windows)")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

CKPT_GLOB = "checkpoint_final_*_seed*_buf*_batch*.pt"

pattern = re.compile(r"checkpoint_final_(?P<env>.+?)_seed(?P<seed>\d+)_buf(?P<buf>\d+)_batch(?P<batch>\d+)\.pt")

rows = []
files = sorted(glob.glob(CKPT_GLOB))
if not files:
    print("No checkpoint_final_* files found in current directory.")
    raise SystemExit(1)

print(f"Found {len(files)} checkpoints. Evaluating each for {args.episodes} episodes...")

for ckpt in files:
    m = pattern.search(ckpt)
    if not m:
        print("Skipping (pattern mismatch):", ckpt)
        continue
    env_name = m.group("env")
    seed = int(m.group("seed"))
    buffer_size = int(m.group("buf"))
    batch_size = int(m.group("batch"))

    print(f"Evaluating {ckpt} -> env={env_name} seed={seed} buf={buffer_size} batch={batch_size}")

    # create env
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    # Assume continuous action spaces (SAC). If discrete, agent needs adjustment.
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n

    # create agent and load actor weights
    agent = SACAgent(obs_dim, act_dim, device=args.device)
    try:
        state = torch.load(ckpt, map_location=args.device)
        agent.actor.load_state_dict(state)
    except Exception as e:
        print("Failed to load checkpoint:", ckpt, e)
        env.close()
        continue

    # run episodes
    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # actor.sample likely returns (action, logprob). adapt if needed
                try:
                    act = agent.actor.sample(obs_t)[0].cpu().numpy()[0]
                except Exception:
                    # fallback: actor(obs) -> logits or action
                    out = agent.actor(obs_t)
                    if isinstance(out, torch.Tensor):
                        # assume direct action output
                        act = out.cpu().numpy()[0]
                    else:
                        raise
            obs, reward, terminated, truncated, _ = env.step(act)
            total += float(reward)
            done = bool(terminated or truncated)
            if args.render:
                env.render()
        rewards.append(total)
    env.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    rows.append({
        "checkpoint": ckpt,
        "env": env_name,
        "seed": seed,
        "buffer": buffer_size,
        "batch": batch_size,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "episodes": args.episodes
    })
    print(f" -> mean={mean_r:.2f}  std={std_r:.2f}")

# Save CSV
df = pd.DataFrame(rows)
df.to_csv("evaluation_results.csv", index=False)
print("Saved evaluation_results.csv with", len(df), "rows.")
