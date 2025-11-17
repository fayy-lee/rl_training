import argparse
import torch
import numpy as np
import gymnasium as gym
import pandas as pd
from sac_utils import SACAgent, ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Hopper-v5", help="Gym environment name")
parser.add_argument("--total_steps", type=int, default=200000, help="Total training steps")
parser.add_argument("--save_at", type=str, default="50000,100000", help="Steps to save checkpoints, comma-separated")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--print_every", type=int, default=1, help="Print progress every N episodes")
args = parser.parse_args()

env = gym.make(args.env)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

agent = SACAgent(obs_dim, act_dim, device="cpu")
buffer = ReplayBuffer(capacity=500000)

save_points = [int(x) for x in args.save_at.split(",")]

episode_rewards = []
obs, _ = env.reset()
ep_reward = 0
step = 0
episode_count = 0

while step < args.total_steps:
    # Convert observation to tensor
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

    # Sample action from actor (no gradients needed)
    with torch.no_grad():
        action = agent.actor.sample(obs_tensor)[0].cpu().numpy()[0]

    # Step environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Store transition
    buffer.push((obs, action, reward, next_obs, float(done)))

    # Update SAC agent
    agent.update(buffer, batch_size=256)

    # Move to next state
    obs = next_obs
    ep_reward += reward
    step += 1

    # End of episode
    if done:
        episode_rewards.append(ep_reward)
        episode_count += 1

        if episode_count % args.print_every == 0:
            print(f"Episode {episode_count} | Step {step} | Reward: {ep_reward}")

        ep_reward = 0
        obs, _ = env.reset()

    # Save checkpoints at specified steps
    if step in save_points:
        path = f"checkpoint_{step}.pt"
        torch.save(agent.actor.state_dict(), path)
        print(f"Saved checkpoint {path}")

torch.save(agent.actor.state_dict(), "checkpoint_final.pt")
pd.DataFrame({"episode_reward": episode_rewards}).to_csv("baseline_rewards.csv", index=False)
print("Training complete. Final model saved as checkpoint_final.pt")
