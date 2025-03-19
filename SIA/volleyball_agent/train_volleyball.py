"""
Training script for Volleyball Agent using CEM
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from env.volleyball_env import VolleyballEnv
from models.cem_model import CEMAgent
from training.train_cem import train_cem, evaluate_agent

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    num_episodes = 1000
    batch_size = 10
    elite_frac = 0.2
    eval_interval = 100
    save_interval = 1000
    
    # Create environment and agent
    env = VolleyballEnv()
    
    # Check if action space is Discrete or Box to handle both cases
    if hasattr(env.action_space, 'n'):
        # Discrete action space
        action_dim = env.action_space.n
    else:
        # Continuous action space
        action_dim = env.action_space.shape[0]
    
    agent = CEMAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,
        population_size=50,
        elite_ratio=0.1,
        noise_std=0.1,
        learning_rate=0.01,
        device=str(device)  # Pass device as string
    )
    
    try:
        # Train the agent
        print("\nStarting training...")
        metrics = train_cem(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            device=device,
            batch_size=batch_size,
            elite_frac=elite_frac,
            eval_interval=eval_interval,
            save_interval=save_interval,
            render=False  # Disable rendering during training
        )
        
        # Plot final results
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['episode_rewards'], label='Episode Reward')
        plt.plot(metrics['mean_rewards'], label='Mean Reward (10 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.close()
        
        # Evaluate the trained agent
        print("\nEvaluating trained agent...")
        eval_metrics = evaluate_agent(env, agent, num_episodes=5, render=True)
        print(f"\nEvaluation Results:")
        print(f"Mean Reward: {eval_metrics['mean_reward']:.2f}")
        print(f"Std Reward: {eval_metrics['std_reward']:.2f}")
        print(f"Min Reward: {eval_metrics['min_reward']:.2f}")
        print(f"Max Reward: {eval_metrics['max_reward']:.2f}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise e
    finally:
        env.close()

if __name__ == "__main__":
    main()