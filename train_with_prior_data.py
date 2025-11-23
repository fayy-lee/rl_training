import argparse
import gymnasium as gym
import pickle
import torch
import pandas as pd
from sac_utils import SACAgent, ReplayBuffer

def train(env_name, dataset_file, total_steps, seed=0, batch_size=256, buffer_capacity=500000, device="cpu"):
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, device=device)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # load dataset (list of transitions)
    with open(dataset_file, "rb") as f:
        data = pickle.load(f)
    for t in data:
        buffer.push(t)
    print(f"Prefilled buffer with {len(data)} transitions from {dataset_file}")

    episode_rewards = []
    obs, _ = env.reset(seed=seed)
    ep_reward = 0
    step = 0

    while step < total_steps:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor, _ = agent.actor.sample(obs_tensor)
        action = action_tensor[0].detach().cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.push((obs, action, float(reward), next_obs, float(done)))
        agent.update(buffer, batch_size=batch_size)

        obs = next_obs
        ep_reward += reward
        step += 1

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

    model_name = f"sac_with_prior_{env_name}_from_{dataset_file.split('.')[0]}_seed{seed}.pt"
    torch.save(agent.actor.state_dict(), model_name)
    reward_csv = model_name.replace(".pt", "_rewards.csv")
    pd.DataFrame({"episode_reward": episode_rewards}).to_csv(reward_csv, index=False)
    print("Finished training with prior data. Saved:", model_name, reward_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_capacity", type=int, default=500000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train(args.env, args.dataset, args.total_steps, seed=args.seed, batch_size=args.batch_size, buffer_capacity=args.buffer_capacity, device=args.device)
