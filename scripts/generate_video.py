# generate_video.py

import argparse
import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo 
from sac_utils import SACAgent # Assumes SACAgent class is available

def generate_video(args):
    """
    Loads a SAC checkpoint and records the agent's performance in the environment.
    """
    
    # --- HELPER FUNCTION (Fixes NameError) ---
    def select_action(agent, obs, device):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std = agent.actor(obs_tensor)
            # Using the mean for evaluation, as per standard practice
            action = torch.tanh(mean)
        return action.cpu().numpy()[0]
    # ----------------------------------------
    
    # Setup paths and device
    video_dir = os.path.join("videos", args.env_name)
    os.makedirs(video_dir, exist_ok=True)
    device = torch.device("cpu") 

    # 1. Create base environment and apply the RecordVideo wrapper
    base_env = gym.make(args.env_name, render_mode="rgb_array")
    
    # The RecordVideo wrapper handles frame capture and video saving upon env.close()
    # Records only the first episode (episode_trigger=lambda x: x == 0)
    env = RecordVideo(
        base_env, 
        video_folder=video_dir, 
        episode_trigger=lambda x: x == 0,
        name_prefix=f"step{args.checkpoint_step}_seed{args.seed}",
        disable_logger=True
    )
    
    # Setup agent dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # 2. Initialize Agent and Load Weights
    agent = SACAgent(obs_dim, act_dim, device=device)

    # Load actor weights from the checkpoint
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.actor.eval() # Set actor to evaluation mode
        print(f"Successfully loaded agent actor from {args.checkpoint_path}")
    except KeyError:
        print("ERROR: Checkpoint file is missing the 'actor' key. Ensure your checkpoint saves the actor state dictionary correctly.")
        return
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {args.checkpoint_path}")
        return

    # 3. Run evaluation 
    for ep in range(args.num_episodes): 
        # The RecordVideo wrapper saves the video file when env.close() is called.
        
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action = select_action(agent, obs, device)
            
            # RecordVideo captures frame on step()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        print(f"Episode {ep} finished. Total Reward: {episode_reward:.2f}")

    # 4. Close the environment to stitch and save the video file
    env.close() 
    base_env.close()
    print(f"Video generation complete. Videos saved to the '{video_dir}' folder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Ant-v4", help="Environment name (e.g., Ant-v4)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Full path to the actor checkpoint file.")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to record.")
    parser.add_argument("--checkpoint_step", type=str, default="300000", help="A label for the checkpoint step.")
    parser.add_argument("--seed", type=int, default=5, help="The seed used for the checkpoint (e.g., 5, 6, 7...).")
    
    args = parser.parse_args()
    generate_video(args)