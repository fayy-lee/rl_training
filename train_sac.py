import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from sac_utils import SACAgent, ReplayBuffer
import random
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Ant-v4", help="Environment name")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--prefill", type=str, default="A", help="Replay buffer type: A=random, B=50k_PPO, C=100k_PPO")
args = parser.parse_args()

# Environment setup
env_name = args.env
env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
device = "cpu"

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)

# Hyperparameters
buffer_size = 100000
batch_size = 256
learning_rate = 3e-4
max_steps = 300000
eval_freq = 5000
learning_starts = 100  # Can set to 0 when prefilled
gamma = 0.99
tau = 0.005

# Initialize agent & buffer
agent = SACAgent(obs_dim, act_dim, device=device, lr=learning_rate, gamma=gamma, tau=tau)
replay_buffer = ReplayBuffer(capacity=buffer_size) # add checkpoint to save the weight of the netwrork at 50k, save weights of final (100k)

# Add select_action method manually (works like .sample)
def select_action(agent, obs, eval_mode=False):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        mean, std = agent.actor(obs_tensor)
        if eval_mode:
            action = torch.tanh(mean)
        else:
            dist = torch.distributions.Normal(mean, std)
            z = dist.sample()
            action = torch.tanh(z)
    return action.cpu().numpy()[0]

agent.select_action = lambda obs, eval_mode=False: select_action(agent, obs, eval_mode)

# Prefill replay buffer
print("Prefilling buffer with random actions...")

prefill_steps = {
    "A": 100000,   # Random fill
    "B": 50000,    # 50K PPO fill
    "C": 100000,   # 100K PPO fill
}[args.prefill]

obs, _ = env.reset()
for step in range(prefill_steps):
    action = env.action_space.sample()
    next_obs, reward, done, truncated, info = env.step(action)
    replay_buffer.push((obs, action, reward, next_obs, float(done)))
    obs = next_obs
    if done or truncated:
        obs, _ = env.reset()

print(f"Replay buffer prefilled with {len(replay_buffer)} transitions.")

# Main SAC training loop
returns = []
episode_reward = 0
obs, _ = env.reset()

for step in range(1, max_steps + 1):
    action = agent.select_action(obs)
    next_obs, reward, done, truncated, info = env.step(action)
    replay_buffer.push((obs, action, reward, next_obs, float(done)))

    episode_reward += reward
    obs = next_obs

    if done or truncated:
        returns.append(episode_reward)
        episode_reward = 0
        obs, _ = env.reset()

    # Update policy
    if step > learning_starts:
        agent.update(replay_buffer, batch_size=batch_size)

    # Periodic logging
    if step % eval_freq == 0:
        avg_return = np.mean(returns[-10:]) if len(returns) > 10 else np.mean(returns)
        print(f"Step: {step} | Avg Return (last 10): {avg_return:.2f}")

# Save learning curve
df = pd.DataFrame(returns, columns=["episode_reward"])
csv_name = f"sac_learning_curve_{env_name}_{args.prefill}_seed{seed}.csv"
df.to_csv(csv_name, index=False)
print(f"Saved learning curve to {csv_name}")

env.close()
