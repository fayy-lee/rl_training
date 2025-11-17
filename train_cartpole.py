import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("CartPole-v1")

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train for 50,000 timesteps
model.learn(total_timesteps=50000)

# Save halfway
model.save("ppo_cartpole_half")

# Train more
model.learn(total_timesteps=50000)

# Save final model
model.save("ppo_cartpole_final")

env.close()
