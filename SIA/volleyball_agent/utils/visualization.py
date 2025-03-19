"""
Utility functions for visualization and training monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import cv2
from IPython.display import display, clear_output
import time

def plot_training_metrics(metrics: Dict[str, List[float]], window_size: int = 100):
    """Plot training metrics with moving average"""
    plt.figure(figsize=(12, 4))
    
    # Plot episode rewards
    plt.subplot(1, 3, 1)
    episode_rewards = np.array(metrics["episode_rewards"])
    plt.plot(episode_rewards, label="Episode Rewards")
    plt.plot(np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid'),
             label=f"Moving Average (n={window_size})")
    plt.title("Episode Rewards")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Reward")
    plt.legend()
    
    # Plot mean rewards
    plt.subplot(1, 3, 2)
    mean_rewards = np.array(metrics["mean_rewards"])
    plt.plot(mean_rewards, label="Mean Rewards")
    plt.plot(np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid'),
             label=f"Moving Average (n={window_size})")
    plt.title("Mean Population Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # Plot elite rewards
    plt.subplot(1, 3, 3)
    elite_rewards = np.array(metrics["elite_rewards"])
    plt.plot(elite_rewards, label="Elite Rewards")
    plt.plot(np.convolve(elite_rewards, np.ones(window_size)/window_size, mode='valid'),
             label=f"Moving Average (n={window_size})")
    plt.title("Elite Population Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def display_episode(env, agent, max_steps: int = 1000):
    """Display a single episode of the agent playing"""
    state = env.reset()
    total_reward = 0
    frames = []
    
    for _ in range(max_steps):
        # Render frame
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Take action
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # Display frames as video
    for frame in frames:
        plt.imshow(frame)
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)
        time.sleep(0.05)
    
    return total_reward

def create_training_video(env, agent, num_episodes: int = 5, output_path: str = "training_video.mp4"):
    """Create a video of the agent playing multiple episodes"""
    state = env.reset()
    frames = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_frames = []
        
        while True:
            frame = env.render(mode='rgb_array')
            episode_frames.append(frame)
            
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            if done:
                break
        
        frames.extend(episode_frames)
    
    # Save video
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release() 