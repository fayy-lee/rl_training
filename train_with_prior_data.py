import argparse
import torch
import numpy as np
import gymnasium as gym
import pandas as pd
from sac_utils import SACAgent, ReplayBuffer

# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Hopper-v5", help="Gym environment name")
parser.add_argument("--prior", required=True, help="Path to prior replay buffer")
parser.add_argument("--total_steps", type=int, default=200000, help="Total training steps")
parser.add_argument("--save_at", type=str, default="50000,100000", help="Steps to save checkpoints")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--print_every", type=int, default=1, help="Print progress every N episodes")
args = parser.parse_args()

# ----------------------------
# Environment setup
# ----------------------------
env = gym.make(args.env)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# ----------------------------
# Agent setup
# ----------------------------
agent = SACAgent(obs_dim, act_dim, device="cpu")

# ----------------------------
# Load prior buffer safely for PyTorch >= 2.6
# ----------------------------
# Use safe_globals to allowlist all types used in your ReplayBuffer
with torch.serialization.safe_globals([
    ReplayBuffer,
    np.ndarray,
    np.float64,
    np.dtype,
    np._core.multiarray._reconstruct
]):
    buffer = torch.load(args.prior, weights_only=False)

print(f"Loaded prior buffer with {len(buffer)} transitions")

# ----------------------------
# Training setup
# ----------------------------
save_points = [int(x) for x in args.save_at.split(",")]

episode_rewards = []
obs, _ = env.reset()
ep_reward = 0
step = 0
episode_count = 0

# ----------------------------
# Training loop
# ----------------------------
while step < args.total_steps:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action = agent.actor.sample(obs_tensor)[0].cpu().numpy()[0]

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Store transition
    buffer.push((obs, action, reward, next_obs, float(done)))

    # Update agent
    agent.update(buffer, batch_size=256)

    obs = next_obs
    ep_reward += reward
    step += 1

    if done:
        episode_rewards.append(ep_reward)
        episode_count += 1

        if episode_count % args.print_every == 0:
            print(f"Episode {episode_count} | Step {step} | Reward: {ep_reward:.2f} | Buffer size: {len(buffer)}")

        ep_reward = 0
        obs, _ = env.reset()

    # Save checkpoints
    if step in save_points:
        path = f"checkpoint_{step}_prior.pt"
        torch.save(agent.actor.state_dict(), path)
        print(f"Saved checkpoint {path}")

# ----------------------------
# Save final model and rewards
# ----------------------------
torch.save(agent.actor.state_dict(), "checkpoint_final_prior.pt")
pd.DataFrame({"episode_reward": episode_rewards}).to_csv("prior_rewards.csv", index=False)
print("Training with prior data complete. Final model saved as checkpoint_final_prior.pt")
