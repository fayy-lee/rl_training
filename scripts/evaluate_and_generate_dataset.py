import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
import numpy as np

# Configuration 
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 100

# Load Models
model_half = PPO.load("ppo_cartpole_half.zip")
model_final = PPO.load("ppo_cartpole_final.zip")

# Create Environment
env = gym.make(ENV_NAME)

def evaluate_model(model, model_name):
    episode_data = []
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        episode_data.append({
            "model": model_name,
            "episode": episode + 1,
            "total_reward": total_reward,
            "steps": steps
        })
        print(f"{model_name} | Episode {episode+1}/{NUM_EPISODES} | Reward: {total_reward}")
    return episode_data

# Evaluate Both Models 
data_half = evaluate_model(model_half, "half")
data_final = evaluate_model(model_final, "final")

# Combine + Save 
df = pd.DataFrame(data_half + data_final)
df.to_csv("cartpole_evaluation_dataset.csv", index=False)
print("\n Dataset saved as cartpole_evaluation_dataset.csv")
print(df.groupby("model")["total_reward"].describe())

env.close()
