import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1️. Simple Offline Dataset Loader
data = pd.read_csv("cartpole_evaluation_dataset.csv")
print(f"Loaded dataset with {len(data)} episodes")

# Synthetic transitions (for RLPD pretraining)
def create_synthetic_transitions(data, env_name="CartPole-v1"):
    env = gym.make(env_name)
    samples = []
    for _, row in data.iterrows():
        obs, _ = env.reset()
        for _ in range(int(row["steps"])):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            samples.append((obs, action, reward, next_obs, terminated or truncated))
            obs = next_obs
            if terminated or truncated:
                break
    env.close()
    return samples

dataset = create_synthetic_transitions(data)
print(f"Synthetic dataset created with {len(dataset)} samples.")

# 2️.  Define Simple Policy Network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3️.  RLPD-style Imitation + Fine-Tuning
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = PolicyNet(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Step 1: Pretrain on offline dataset 
epochs = 10
batch_size = 64
rewards_per_epoch = []

for epoch in range(epochs):
    np.random.shuffle(dataset)
    losses = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        obs = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)

        logits = policy(obs)
        loss = loss_fn(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses)
    print(f"Epoch {epoch+1}/{epochs} | Offline Pretrain Loss: {avg_loss:.4f}")
    rewards_per_epoch.append(-avg_loss)

# Step 2: Online Fine-Tuning 
def evaluate_policy(env, policy, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            logits = policy(obs_tensor)
            action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        total_rewards.append(total_reward)
    return np.mean(total_rewards), total_rewards

fine_tune_rewards = []
all_fine_tune_rewards = []  # store individual episode rewards

for step in range(10):
    avg_reward, per_episode_rewards = evaluate_policy(env, policy, episodes=10)
    fine_tune_rewards.append(avg_reward)
    all_fine_tune_rewards.extend(per_episode_rewards)
    print(f"Fine-tuning Step {step+1}/10 | Avg Reward: {avg_reward:.2f}")

# Step 3:  Save RLPD Rewards to CSV
df_rewards = pd.DataFrame({"episode": list(range(1, len(all_fine_tune_rewards)+1)),
                           "reward": all_fine_tune_rewards})
df_rewards.to_csv("rlpd_cartpole_learning_curve.csv", index=False)
print("RLPD episode rewards saved to 'rlpd_cartpole_learning_curve.csv'")

# 4.  Plot Learning Curve
plt.figure(figsize=(8,5))
plt.plot(rewards_per_epoch, label="Offline Pretrain (synthetic)")
plt.plot(range(len(rewards_per_epoch), len(rewards_per_epoch)+len(fine_tune_rewards)),
         fine_tune_rewards, label="Online Fine-tune (RLPD-style)")
plt.xlabel("Epoch / Step")
plt.ylabel("Reward / -Loss Proxy")
plt.legend()
plt.title("RLPD Training Curve - CartPole")
plt.tight_layout()
plt.savefig("rlpd_cartpole_learning_curve.png")
plt.show()

env.close()
