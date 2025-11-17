import argparse
import torch
import numpy as np
import gymnasium as gym
from sac_utils import SACAgent, ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Hopper-v5", help="Gym environment name")
parser.add_argument("--checkpoint", required=True, help="Path to saved actor checkpoint")
parser.add_argument("--target_transitions", type=int, default=5200, help="Number of transitions to collect")
parser.add_argument("--save_buffer", default="prior_buffer.pt", help="Path to save collected buffer")
args = parser.parse_args()

# Environment setup
env = gym.make(args.env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Agent and buffer
agent = SACAgent(obs_dim, act_dim, device="cpu")
agent.actor.load_state_dict(torch.load(args.checkpoint))
agent.actor.eval()

buffer = ReplayBuffer(capacity=500000)

# Collect transitions until target reached
total_steps = 0
episode_count = 0

while len(buffer) < args.target_transitions:
    obs, _ = env.reset()
    ep_reward = 0
    done = False
    episode_count += 1

    while not done and len(buffer) < args.target_transitions:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = agent.actor.sample(obs_tensor)[0].cpu().numpy()[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        ep_reward += reward
        total_steps += 1

    print(f"Collected Episode {episode_count} | Reward: {ep_reward} | Total transitions: {len(buffer)}")

# Save replay buffer
torch.save(buffer, args.save_buffer)
print(f"Collected buffer with {len(buffer)} transitions saved to {args.save_buffer}")
