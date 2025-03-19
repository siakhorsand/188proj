"""
Improved training script for SlimeVolley using CEM agent
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from typing import List, Dict

def train_cem(
    env,
    agent,
    num_episodes: int,
    device=None,
    batch_size: int = 20,  # Increased from 10
    elite_frac: float = 0.2,
    eval_interval: int = 50,
    save_interval: int = 100,
    save_dir: str = "saved_models",
    render: bool = False,
    reward_shaping: bool = True  # Enable reward shaping
) -> Dict[str, List[float]]:
    """
    Train the CEM agent with improved training process
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cem_volleyball.pt")
    best_model_path = os.path.join(save_dir, "cem_volleyball_best.pt")
    
    # Handle device properly
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Ensure agent is using the correct device
    agent.device = device
    
    metrics = {
        "episode_rewards": [],
        "mean_rewards": [],
        "std_rewards": [],
        "elite_rewards": [],
        "best_reward": float("-inf"),
        "training_duration": 0,
        "noise_std": []
    }
    
    start_time = time.time()
    
    # Initialize the plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Main reward plot
    rewards_line, = ax1.plot([], [], 'b-', alpha=0.3, label='Episode Reward')
    mean_line, = ax1.plot([], [], 'r-', linewidth=2, label='Mean Reward (10 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Noise decay plot
    noise_line, = ax2.plot([], [], 'g-', label='Noise Standard Deviation')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Noise Std')
    ax2.set_title('Exploration Noise Decay')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Setup progress bar for entire training
    pbar = tqdm(total=num_episodes, desc="Training")
    
    # For tracking best performing agent
    best_mean_reward = float('-inf')
    
    try:
        for batch_num in range(num_episodes):
            # Collect batch of episodes
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_episode_rewards = []
            batch_step_counts = []
            
            for batch_idx in range(batch_size):
                states = []
                actions = []
                rewards = []
                shaped_rewards = []
                state = env.reset()
                done = False
                episode_reward = 0
                step_count = 0
                
                # For reward shaping
                prev_ball_y = None
                prev_ball_x = None
                prev_agent_x = None
                total_movement = 0
                hit_count = 0
                
                while not done:
                    # Process state 
                    if isinstance(state, (int, float)):
                        state = np.array([state], dtype=np.float32)
                    elif not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)
                    
                    states.append(state)
                    
                    # Select action
                    action = agent.select_action(state)
                    actions.append(action)
                    
                    # Step environment
                    next_state, reward, done, info = env.step(action)
                    
                    # Extract ball and agent position for reward shaping
                    if reward_shaping and len(state) >= 12:  # Make sure we have enough state dimensions
                        # In SlimeVolley state: [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy]
                        ball_x = state[4]
                        ball_y = state[5]
                        agent_x = state[0]
                        
                        shaped_reward = reward  # Start with the original reward
                        
                        # 1. Reward for ball going higher (keeping ball in play)
                        if prev_ball_y is not None and ball_y > prev_ball_y:
                            shaped_reward += 0.01
                            
                        # 2. Reward for getting the ball over to opponent side
                        if prev_ball_x is not None and prev_ball_x <= 0 and ball_x > 0:
                            shaped_reward += 0.2
                            
                        # 3. Small penalty for not moving (encourage exploration)
                        if prev_agent_x is not None:
                            movement = abs(agent_x - prev_agent_x)
                            total_movement += movement
                            if movement < 0.01:  # Almost no movement
                                shaped_reward -= 0.01
                            
                        # 4. Reward for hitting the ball
                        if prev_ball_y is not None and abs(ball_x - agent_x) < 1.5 and abs(ball_y - 1.5) < 1.5:
                            if hit_count == 0:  # Only reward the first hit per rally
                                shaped_reward += 0.3
                                hit_count += 1
                        
                        # Store for next comparison
                        prev_ball_y = ball_y
                        prev_ball_x = ball_x
                        prev_agent_x = agent_x
                        
                        rewards.append(shaped_reward)
                    else:
                        rewards.append(reward)
                    
                    episode_reward += reward
                    
                    state = next_state
                    step_count += 1
                    
                    # Reset hit count if point is scored
                    if reward != 0:
                        hit_count = 0
                
                # Store episode data
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_episode_rewards.append(episode_reward)
                batch_step_counts.append(step_count)
            
            # Convert actions to numpy array first to avoid PyTorch warning
            batch_actions_np = np.array(batch_actions, dtype=np.float32)
            
            # Update the agent
            update_info = agent.update(batch_states, batch_actions_np, batch_rewards, elite_frac)
            
            # Calculate and store metrics 
            batch_mean = np.mean(batch_episode_rewards)
            batch_std = np.std(batch_episode_rewards)
            
            metrics['episode_rewards'].extend(batch_episode_rewards)
            
            # Calculate recent mean (over last 10 episodes)
            recent_rewards = metrics['episode_rewards'][-10:]
            current_mean = np.mean(recent_rewards)
            current_std = np.std(recent_rewards)
            
            metrics['mean_rewards'].append(current_mean)
            metrics['std_rewards'].append(current_std)
            metrics['elite_rewards'].append(update_info['elite_reward'])
            metrics['noise_std'].append(update_info['noise_std'])
            
            # Update best reward
            if current_mean > best_mean_reward:
                best_mean_reward = current_mean
                agent.save(best_model_path)
                print(f"\nNew best model saved with mean reward: {best_mean_reward:.2f}")
            
            metrics['best_reward'] = max(metrics['best_reward'], max(batch_episode_rewards))
            
            # Update progress bar
            pbar.set_postfix({
                'mean_reward': f"{batch_mean:.2f}",
                'std': f"{batch_std:.2f}",
                'best': f"{metrics['best_reward']:.2f}",
                'noise': f"{update_info['noise_std']:.3f}"
            })
            pbar.update(1)
            
            # Update the plots
            if batch_num % 1 == 0:  # Update every batch
                # Main reward plot
                episodes_x = list(range(len(metrics['episode_rewards'])))
                rewards_line.set_data(episodes_x, metrics['episode_rewards'])
                
                # Mean reward line (plotted at each batch)
                mean_x = list(range(0, len(metrics['episode_rewards']), batch_size))
                if len(mean_x) < len(metrics['mean_rewards']):
                    mean_x.append(len(metrics['episode_rewards']) - 1)
                mean_line.set_data(mean_x[:len(metrics['mean_rewards'])], metrics['mean_rewards'])
                
                # Noise decay plot
                noise_x = list(range(len(metrics['noise_std'])))
                noise_line.set_data(noise_x, metrics['noise_std'])
                
                # Adjust axes
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # Perform evaluation periodically
            if (batch_num + 1) % eval_interval == 0:
                eval_rewards = []
                for _ in range(5):  # Evaluate on 5 episodes
                    eval_state = env.reset()
                    eval_done = False
                    eval_episode_reward = 0
                    
                    while not eval_done:
                        if isinstance(eval_state, (int, float)):
                            eval_state = np.array([eval_state], dtype=np.float32)
                        elif not isinstance(eval_state, np.ndarray):
                            eval_state = np.array(eval_state, dtype=np.float32)
                        
                        # Use the best parameters found so far (no exploration noise)
                        with torch.no_grad():
                            eval_action = agent.select_action(eval_state)
                            # Remove exploration noise for evaluation
                            if hasattr(agent, 'policy'):
                                agent.policy.set_params(agent.best_params if agent.best_params is not None else agent.mean)
                                eval_state_tensor = torch.from_numpy(eval_state).float().to(agent.device)
                                eval_action = agent.policy(eval_state_tensor).cpu().numpy()
                            
                        eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_state = eval_next_state
                    
                    eval_rewards.append(eval_episode_reward)
                
                eval_mean = np.mean(eval_rewards)
                print(f"\nEvaluation at episode {batch_num + 1}: Mean reward = {eval_mean:.2f}")
            
            # Save model periodically
            if (batch_num + 1) % save_interval == 0:
                agent.save(save_path)
                print(f"\nModel saved to {save_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"\nTraining interrupted due to error: {str(e)}")
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        raise e
    
    finally:
        pbar.close()
        plt.ioff()
        metrics['training_duration'] = time.time() - start_time
        env.close()
        
    return metrics

def evaluate_agent(env, agent, num_episodes: int = 10, render: bool = True) -> Dict[str, float]:
    """
    Evaluate the trained agent
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Use best parameters for evaluation
        if hasattr(agent, 'best_params') and agent.best_params is not None:
            agent.policy.set_params(agent.best_params)
        
        while not done:
            if isinstance(state, (int, float)):
                state = np.array([state], dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Get action without exploration noise
            with torch.no_grad():
                if hasattr(agent, 'policy'):
                    state_tensor = torch.from_numpy(state).float().to(agent.device)
                    action = agent.policy(state_tensor).cpu().numpy()
                else:
                    action = agent.select_action(state)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            state = next_state
            
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        print(f"Evaluation episode {episode+1}: Reward = {episode_reward:.2f}")
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards)
    }