import argparse
import torch
import gymnasium as gym
import pickle
from sac_utils import SACAgent

def collect(env_name, ckpt_file, output_file, rollout_steps, seed=0, device="cpu"):
    env = gym.make(env_name)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, device=device)
    state = torch.load(ckpt_file, map_location=device)
    agent.actor.load_state_dict(state)

    dataset = []
    obs, _ = env.reset()

    for i in range(rollout_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor, _ = agent.actor.sample(obs_tensor)
        action = action_tensor[0].detach().cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        dataset.append((obs, action, float(reward), next_obs, float(done)))
        obs = next_obs
        if done:
            obs, _ = env.reset()

    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} transitions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--rollout_steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    collect(args.env, args.ckpt, args.out, args.rollout_steps, seed=args.seed, device=args.device)
