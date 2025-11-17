import argparse
import torch
import numpy as np
import gymnasium as gym
from sac_utils import SACAgent

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Hopper-v5", help="Gym environment name")
parser.add_argument("--checkpoint", required=True, help="Path to trained actor checkpoint")
parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate")
parser.add_argument("--render", action="store_true", help="Render environment")
args = parser.parse_args()

# Environment setup
env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Load trained agent
agent = SACAgent(obs_dim, act_dim, device="cpu")
agent.actor.load_state_dict(torch.load(args.checkpoint))
agent.actor.eval()

# Evaluation loop
all_rewards = []

for ep in range(1, args.episodes + 1):
    obs, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = agent.actor.sample(obs_tensor)[0].cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        obs = next_obs

        if args.render:
            env.render()

    all_rewards.append(ep_reward)
    print(f"Episode {ep} | Reward: {ep_reward:.2f}")

env.close()

# Summary
mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)
print(f"\nEvaluation over {args.episodes} episodes: mean={mean_reward:.2f}, std={std_reward:.2f}")
