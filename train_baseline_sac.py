import argparse
import torch
import numpy as np
import gymnasium as gym
import pandas as pd
from sac_utils import SACAgent, ReplayBuffer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--save_at", type=str, default="50000,100000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env = gym.make(args.env)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    # action can be Box (continuous)
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, device=args.device)
    buffer = ReplayBuffer(capacity=args.buffer_size)

    save_points = [int(x) for x in args.save_at.split(",") if x.strip()!='']

    episode_rewards = []
    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0
    step = 0

    print(f"Starting training: env={args.env} seed={args.seed} buffer={args.buffer_size} batch={args.batch_size}")

    while step < args.total_steps:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
        action_tensor, _ = agent.actor.sample(obs_tensor)
        action = action_tensor[0].detach().cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        buffer.push((obs, action, float(reward), next_obs, float(done)))

        # only update once buffer has some samples
        agent.update(buffer, batch_size=args.batch_size)

        obs = next_obs
        ep_reward += reward
        step += 1

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

        if step in save_points:
            ckpt_name = f"checkpoint_{step}.pt"
            torch.save(agent.actor.state_dict(), ckpt_name)
            print(f"[step {step}] Saved checkpoint {ckpt_name}")

    # final save
    final_name = f"checkpoint_final_{args.env}_seed{args.seed}_buf{args.buffer_size}_batch{args.batch_size}.pt"
    torch.save(agent.actor.state_dict(), final_name)
    reward_csv = f"baseline_rewards_{args.env}_seed{args.seed}_buf{args.buffer_size}_batch{args.batch_size}.csv"
    pd.DataFrame({"episode_reward": episode_rewards}).to_csv(reward_csv, index=False)
    print("Training complete. Saved:", final_name, reward_csv)

if __name__ == "__main__":
    main()
