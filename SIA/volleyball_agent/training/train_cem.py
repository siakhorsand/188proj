"""
Training script for Cross Entropy Method (CEM) agent
Includes standard training with reward shaping and self-play training
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from typing import List, Tuple, Dict
import copy
import warnings

def train_cem(
    env,
    agent,
    num_episodes: int,
    device=None,
    batch_size: int = 20,
    elite_frac: float = 0.2,
    eval_interval: int = 20,
    save_interval: int = 50,
    save_dir: str = "saved_models",
    render: bool = False
) -> Dict[str, List[float]]:
    """
    Train the CEM agent with volleyball-specific reward shaping
    
    Args:
        env: The volleyball environment
        agent: The CEM agent
        num_episodes: Number of episodes to train for
        device: The device to use for training (cpu, cuda, or mps)
        batch_size: Number of episodes to collect before updating
        elite_frac: Fraction of episodes to use as elite samples
        eval_interval: How often to evaluate the agent
        save_interval: How often to save the model
        save_dir: Directory to save models in
        render: Whether to render the environment during training
    
    Returns:
        Dictionary containing training metrics
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
    agent.mean = agent.mean.to(device)
    agent.std = agent.std.to(device)
    agent.population = agent.population.to(device)
    
    metrics = {
        "episode_rewards": [],
        "mean_rewards": [],
        "std_rewards": [],
        "elite_rewards": [],
        "best_reward": float("-inf"),
        "training_duration": 0
    }
    
    start_time = time.time()
    
    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    rewards_line, = ax.plot([], [], 'b-', alpha=0.3, label='Episode Reward')
    mean_line, = ax.plot([], [], 'r-', linewidth=2, label='Mean Reward (10 episodes)')
    std_upper, = ax.plot([], [], 'g:', alpha=0.5)
    std_lower, = ax.plot([], [], 'g:', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    # Setup progress bar for entire training
    pbar = tqdm(total=num_episodes, desc="Training")
    
    # For tracking best performing agent
    best_mean_reward = float('-inf')
    best_params = None
    
    try:
        for batch_num in range(num_episodes):
            # Collect batch of episodes
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_episode_rewards = []  # Store total episode rewards
            
            for batch_idx in range(batch_size):
                states = []
                actions = []
                rewards = []
                shaped_rewards = []  # For reward shaping
                state = env.reset()
                done = False
                episode_reward = 0
                
                # Slime volley specific tracking
                prev_ball_x = None
                prev_ball_y = None
                hit_ball = False
                crossed_net = False
                
                step_count = 0
                while not done:
                    # Process state
                    if isinstance(state, (int, float)):
                        state = np.array([state], dtype=np.float32)
                    elif not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)
                    
                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    
                    # ---- REWARD SHAPING FOR VOLLEYBALL ----
                    shaped_reward = reward  # Start with the original reward
                    
                    # Extract information for reward shaping if we have the full observation
                    if len(state) >= 12:
                        # In SlimeVolley state: [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy]
                        agent_x = state[0]
                        agent_y = state[1]
                        ball_x = state[4]
                        ball_y = state[5]
                        ball_vx = state[6]
                        ball_vy = state[7]
                        
                        # 1. Reward for being near the ball horizontally
                        dist_to_ball_x = abs(agent_x - ball_x)
                        if dist_to_ball_x < 2.0:
                            shaped_reward += 0.01
                        
                        # 2. Reward for hitting the ball upward
                        if prev_ball_y is not None:
                            # Detect if ball velocity changed significantly (likely a hit)
                            if ball_y > prev_ball_y and ball_vy > 0 and dist_to_ball_x < 2.0:
                                shaped_reward += 0.3
                                hit_ball = True
                        
                        # 3. BIG reward for getting ball across the net
                        if prev_ball_x is not None:
                            # Ball crossed from agent side to opponent side
                            if prev_ball_x < 0 and ball_x > 0:
                                shaped_reward += 1.0  # Significant reward for crossing net
                                crossed_net = True
                        
                        # 4. Small reward for moving toward the ball
                        if prev_ball_x is not None:
                            # If ball is moving toward agent
                            if ball_x < 0 and ball_vx < 0:
                                # Reward moving in direction of ball
                                if (agent_x < ball_x and action[0] > 0) or (agent_x > ball_x and action[1] > 0):
                                    shaped_reward += 0.05
                        
                        # 5. Small reward for jumping when ball is above
                        if ball_y > agent_y and action[2] > 0:
                            shaped_reward += 0.05
                            
                        # Store current positions for next step comparison
                        prev_ball_x = ball_x
                        prev_ball_y = ball_y
                    
                    # Add original reward for terminal states (point scored)
                    if reward != 0:
                        # If agent scored a point, give them the full reward
                        if reward > 0:
                            shaped_reward = reward
                        # If agent lost, penalize more if they never hit the ball or got it over net
                        elif reward < 0:
                            if not hit_ball:
                                shaped_reward = reward * 1.5  # Bigger penalty for not hitting
                            elif not crossed_net:
                                shaped_reward = reward * 1.2  # Smaller penalty for not crossing net
                        
                        # Reset tracking for new point
                        hit_ball = False
                        crossed_net = False
                    
                    states.append(state)
                    actions.append(action)
                    rewards.append(shaped_reward)  # Use shaped reward
                    
                    episode_reward += reward  # Keep original reward for metrics
                    
                    state = next_state
                    step_count += 1
                
                # Store episode data
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_episode_rewards.append(episode_reward)
            
            # Convert actions to numpy array first to avoid PyTorch warning
            batch_actions_np = np.array(batch_actions, dtype=np.float32)
            
            # Update the agent
            update_info = agent.update(batch_states, batch_actions_np, batch_rewards, elite_frac)
            
            # Store metrics
            metrics['episode_rewards'].extend(batch_episode_rewards)
            
            # Calculate and store running metrics
            start_idx = max(0, len(metrics['episode_rewards']) - 10)
            recent_rewards = metrics['episode_rewards'][start_idx:]
            current_mean = np.mean(recent_rewards)
            current_std = np.std(recent_rewards)
            
            metrics['mean_rewards'].append(current_mean)
            metrics['std_rewards'].append(current_std)
            metrics['elite_rewards'].append(update_info.get('elite_reward', 0))
            
            # Track best parameters
            if current_mean > best_mean_reward:
                best_mean_reward = current_mean
                best_params = agent.mean.clone()
                torch.save({
                    'mean': best_params,
                    'std': agent.std,
                    'reward': best_mean_reward
                }, best_model_path)
                print(f"\nNew best model saved with mean reward: {best_mean_reward:.2f}")
            
            # Update best reward
            metrics['best_reward'] = max(metrics['best_reward'], max(batch_episode_rewards))
            
            # Update progress bar
            batch_mean = np.mean(batch_episode_rewards)
            batch_std = np.std(batch_episode_rewards)
            pbar.set_postfix({
                'mean_reward': f"{batch_mean:.2f}",
                'std': f"{batch_std:.2f}",
                'best': f"{metrics['best_reward']:.2f}"
            })
            pbar.update(1)
            
            # Update the plot
            episodes_x = list(range(len(metrics['episode_rewards'])))
            rewards_line.set_data(episodes_x, metrics['episode_rewards'])
            
            mean_episodes = list(range(0, len(metrics['episode_rewards']), batch_size))
            if len(mean_episodes) < len(metrics['mean_rewards']):
                mean_episodes.append(len(metrics['episode_rewards']) - 1)
                
            mean_line.set_data(mean_episodes[:len(metrics['mean_rewards'])], metrics['mean_rewards'])
            
            # Update standard deviation bands
            if len(metrics['std_rewards']) > 0:
                std_upper.set_data(
                    mean_episodes[:len(metrics['mean_rewards'])], 
                    [m + s for m, s in zip(metrics['mean_rewards'], metrics['std_rewards'])]
                )
                std_lower.set_data(
                    mean_episodes[:len(metrics['mean_rewards'])], 
                    [m - s for m, s in zip(metrics['mean_rewards'], metrics['std_rewards'])]
                )
            
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
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
        
        # Before closing, load best parameters if available
        if best_params is not None:
            agent.mean = best_params
        
        env.close()
        
    return metrics

def evaluate_agent(env, agent, num_episodes: int = 10, render: bool = True) -> Dict[str, float]:
    """
    Evaluate the trained agent
    
    Args:
        env: The volleyball environment
        agent: The trained CEM agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    
    Returns:
        Dictionary containing evaluation metrics
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if isinstance(state, (int, float)):
                state = np.array([state], dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Use the agent's policy without exploration noise for evaluation
            with torch.no_grad():
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

def train_selfplay_cem(
    env,
    agent,
    num_episodes: int,
    device=None,
    batch_size: int = 20,
    elite_frac: float = 0.2,
    eval_interval: int = 50,
    save_interval: int = 100,
    save_dir: str = "saved_models",
    render: bool = False
) -> Dict[str, List[float]]:
    """
    Train CEM agent using self-play in SlimeVolley
    
    Args:
        env: SlimeVolley environment (needs to be multiagent compatible)
        agent: The CEM agent for right player
        num_episodes: Number of episodes to train for
        device: Device to use for tensor operations
        batch_size: Episodes per batch before update
        elite_frac: Fraction of elite samples
        eval_interval: Episodes between evaluations
        save_interval: Episodes between model saves
        save_dir: Directory to save models to
        render: Whether to render training
        
    Returns:
        Dictionary of training metrics
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
    
    # Create second agent for self-play (initially identical)
    opponent_agent = copy.deepcopy(agent)
    
    # Ensure both agents are using the correct device
    agent.device = device
    agent.mean = agent.mean.to(device)
    agent.std = agent.std.to(device)
    agent.population = agent.population.to(device)
    
    opponent_agent.device = device
    opponent_agent.mean = opponent_agent.mean.to(device)
    opponent_agent.std = opponent_agent.std.to(device)
    opponent_agent.population = opponent_agent.population.to(device)
    
    # Historical versions of the agent to train against (for diversity)
    hall_of_fame = []
    
    metrics = {
        "episode_rewards": [],        # Rewards for the main agent
        "opponent_rewards": [],       # Rewards for the opponent agent
        "mean_rewards": [],           # Moving average of main agent rewards
        "opponent_mean_rewards": [],  # Moving average of opponent rewards
        "vs_baseline_rewards": [],    # Occasional evaluation vs built-in policy
        "std_rewards": [],            # Standard deviation of rewards
        "best_reward": float("-inf"),
        "training_duration": 0
    }
    
    start_time = time.time()
    
    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    rewards_line, = ax.plot([], [], 'b-', alpha=0.3, label='Agent Reward')
    opponent_line, = ax.plot([], [], 'r-', alpha=0.3, label='Opponent Reward')
    mean_line, = ax.plot([], [], 'b-', linewidth=2, label='Agent Mean Reward')
    op_mean_line, = ax.plot([], [], 'r-', linewidth=2, label='Opponent Mean Reward')
    eval_line, = ax.plot([], [], 'g*', markersize=8, label='Vs Built-in Policy')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Self-Play Training Progress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    # Setup progress bar for entire training
    pbar = tqdm(total=num_episodes, desc="Self-Play Training")
    
    # For tracking best performing agent
    best_mean_reward = float('-inf')
    best_params = None
    
    try:
        for episode in range(num_episodes):
            # Decide if we should use a historical opponent for this episode
            use_random_opponent = False
            if len(hall_of_fame) > 0 and np.random.random() < 0.3:  # 30% chance to use historical opponent
                use_random_opponent = True
                historical_opponent = np.random.choice(hall_of_fame)
                historical_mean = historical_opponent['mean']
                opponent_agent.mean = historical_mean.clone()
            
            # Collect episode data
            states_right = []
            actions_right = []
            rewards_right = []
            
            states_left = []
            actions_left = []
            rewards_left = []
            
            # Initialize environment
            state_right = env.reset()
            # For SlimeVolley, the observation from the left agent's perspective is in info['otherObs']
            _, _, _, info = env.step(np.zeros(3))  # Take a dummy step to get otherObs
            state_left = info['otherObs']
            
            done = False
            episode_reward_right = 0
            episode_reward_left = 0
            

            # Initialize tracking variables for reward shaping
            prev_ball_x = None
            prev_ball_y = None
            prev_ball_vy = None
            hit_count = 0
            successful_hit = False
            crossed_net = False
            agent_move_counter = 0
            good_position = False
            # Play one episode
            while not done:
                # Process state for right agent
                if isinstance(state_right, (int, float)):
                    state_right = np.array([state_right], dtype=np.float32)
                elif not isinstance(state_right, np.ndarray):
                    state_right = np.array(state_right, dtype=np.float32)
                
                # Process state for left agent
                if isinstance(state_left, (int, float)):
                    state_left = np.array([state_left], dtype=np.float32)
                elif not isinstance(state_left, np.ndarray):
                    state_left = np.array(state_left, dtype=np.float32)
                
                # Select actions for both agents
                action_right = agent.select_action(state_right)
                action_left = opponent_agent.select_action(state_left)
                
                # Take step in environment with both actions
                next_state_right, reward_right, done, info = env.step(action_right, action_left)
                next_state_left = info['otherObs']
                reward_left = -reward_right  # Rewards are opposite in zero-sum game
                
                # Store transitions
                states_right.append(state_right)
                actions_right.append(action_right)
                rewards_right.append(reward_right)
                
                states_left.append(state_left)
                actions_left.append(action_left)
                rewards_left.append(reward_left)
                # Enhanced reward shaping for volleyball skills
            if len(state_right) >= 12:  # Make sure we have full observation
                # Extract positions from state
                agent_x = state_right[0]
                agent_y = state_right[1]
                ball_x = state_right[4]
                ball_y = state_right[5]
                ball_vx = state_right[6]
                ball_vy = state_right[7]

                # 1. Reward for being in a good position - near where the ball will land
                if ball_x < 0 and ball_vx < 0:  # Ball moving toward agent
                    # Predict approximately where ball will land
                    time_to_ground = max(0, (ball_y - 1.5) / -9.8)  # Simple physics
                    predicted_x = ball_x + ball_vx * time_to_ground

                    # Distance to predicted landing spot
                    distance_to_landing = abs(agent_x - predicted_x)
                    if distance_to_landing < 2.0:
                        rewards_right[-1] += 0.2  # Significant reward for good positioning
                        good_position = True

                # 2. Bigger reward for hitting the ball successfully
                if prev_ball_y is not None and prev_ball_vy is not None:
                    # Ball was going down but now going up - likely a hit
                    if prev_ball_vy < 0 and ball_vy > 0 and abs(agent_x - ball_x) < 2.0:
                        rewards_right[-1] += 0.5
                        successful_hit = True
                        hit_count += 1

                        # Extra reward for multiple successful hits in a rally
                        if hit_count > 1:
                            rewards_right[-1] += 0.2 * hit_count  # Scales with number of hits

                # 3. Much bigger reward for getting the ball over the net
                if prev_ball_x is not None and prev_ball_x < 0 and ball_x > 0:
                    rewards_right[-1] += 1.5  # Very significant reward
                    crossed_net = True

                    # Even more reward if it was intentional (after a successful hit)
                    if successful_hit:
                        rewards_right[-1] += 0.5

                # 4. Reward for active movement and jumping at appropriate times
                moved_this_step = False
                if action_right[0] > 0 or action_right[1] > 0:
                    moved_this_step = True

                # Reward jumping when ball is nearby and above
                if action_right[2] > 0 and ball_y > agent_y and abs(ball_x - agent_x) < 2.0:
                    rewards_right[-1] += 0.15

                # Slightly penalize inactivity
                if not moved_this_step:
                    agent_move_counter += 1
                    if agent_move_counter > 10:  # Not moving for many steps
                        rewards_right[-1] -= 0.01 * min(agent_move_counter, 20)  # Cap the penalty
                else:
                    agent_move_counter = 0

                # Store current positions for next step comparison
                prev_ball_x = ball_x
                prev_ball_y = ball_y
                prev_ball_vy = ball_vy

            # Reset tracking variables when a point is scored
            if reward_right != 0:
                hit_count = 0
                successful_hit = False
                crossed_net = False
                agent_move_counter = 0
                good_position = False

            # Apply symmetrical reward shaping to opponent (flip the signs)
            rewards_left[-1] = -rewards_right[-1]
                # Update episode rewards
            episode_reward_right += reward_right
            episode_reward_left += reward_left
                
                # Update states
            state_right = next_state_right
            state_left = next_state_left
            if render:
                env.render()
            
            # Store episode rewards
            metrics['episode_rewards'].append(episode_reward_right)
            metrics['opponent_rewards'].append(episode_reward_left)
            
            # Batch complete - update both agents
            if (episode + 1) % batch_size == 0:
                # Convert actions to numpy arrays
                actions_right_np = np.array(actions_right, dtype=np.float32)
                actions_left_np = np.array(actions_left, dtype=np.float32)
                
                # Update agents
                update_info_right = agent.update(states_right, actions_right_np, rewards_right, elite_frac)
                if not use_random_opponent:
                    update_info_left = opponent_agent.update(states_left, actions_left_np, rewards_left, elite_frac)


                if hasattr(agent, 'noise_std'):
                    agent.noise_std *= 0.998  # Very slow decay
                    agent.noise_std = max(agent.noise_std, 0.05)

                if hasattr(opponent_agent, 'noise_std') and not use_random_opponent:
                    opponent_agent.noise_std *= 0.998  # Apply same decay to opponent
                    opponent_agent.noise_std = max(opponent_agent.noise_std, 0.05)
                start_idx = max(0, len(metrics['episode_rewards']) - 10)
                recent_rewards = metrics['episode_rewards'][start_idx:]
                recent_op_rewards = metrics['opponent_rewards'][start_idx:]
                
                current_mean = np.mean(recent_rewards)
                current_op_mean = np.mean(recent_op_rewards)
                current_std = np.std(recent_rewards)
                
                metrics['mean_rewards'].append(current_mean)
                metrics['opponent_mean_rewards'].append(current_op_mean)
                metrics['std_rewards'].append(current_std)
                
                # Update best reward and save best model
                if current_mean > best_mean_reward:
                    best_mean_reward = current_mean
                    best_params = agent.mean.clone()
                    torch.save({
                        'mean': best_params,
                        'std': agent.std,
                        'reward': best_mean_reward
                    }, best_model_path)
                    print(f"\nNew best model saved with mean reward: {best_mean_reward:.2f}")
                
                # Add current version to hall of fame (occasionally)
                if len(hall_of_fame) < 5 or np.random.random() < 0.1:  # 10% chance after hall of fame has 5 models
                    hall_of_fame.append({
                        'mean': agent.mean.clone(),
                        'performance': current_mean
                    })
                    print(f"Added new model to hall of fame (size: {len(hall_of_fame)})")
                
                # If hall of fame gets too big, remove worst performing model
                if len(hall_of_fame) > 10:
                    hall_of_fame.sort(key=lambda x: x['performance'])
                    hall_of_fame.pop(0)  # Remove worst performer
                
                # Update progress bar
                pbar.set_postfix({
                    'agent_reward': f"{current_mean:.2f}",
                    'opp_reward': f"{current_op_mean:.2f}",
                    'best': f"{best_mean_reward:.2f}"
                })
            
            pbar.update(1)
            
            # Update the plot
            if (episode + 1) % 5 == 0:
                episodes_x = list(range(len(metrics['episode_rewards'])))
                rewards_line.set_data(episodes_x, metrics['episode_rewards'])
                opponent_line.set_data(episodes_x, metrics['opponent_rewards'])
                
                mean_x = list(range(0, len(metrics['episode_rewards']), batch_size))
                if len(mean_x) < len(metrics['mean_rewards']):
                    mean_x.append(len(metrics['episode_rewards']) - 1)
                
                mean_line.set_data(mean_x[:len(metrics['mean_rewards'])], metrics['mean_rewards'])
                op_mean_line.set_data(mean_x[:len(metrics['opponent_mean_rewards'])], metrics['opponent_mean_rewards'])
                
                if len(metrics['vs_baseline_rewards']) > 0:
                    eval_x = list(range(0, len(metrics['episode_rewards']), eval_interval))[:len(metrics['vs_baseline_rewards'])]
                    eval_line.set_data(eval_x, metrics['vs_baseline_rewards'])
                
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # Evaluate against built-in opponent periodically
            if (episode + 1) % eval_interval == 0:
                # Reset the environment to ensure we're using the baseline opponent
                env.reset()
                
                # Evaluate for 5 episodes against built-in opponent
                eval_rewards = []
                for _ in range(5):
                    eval_state = env.reset()
                    eval_done = False
                    eval_episode_reward = 0
                    
                    # Use the best parameters found so far for evaluation
                    if best_params is not None:
                        agent.mean = best_params.clone()
                    
                    while not eval_done:
                        if isinstance(eval_state, (int, float)):
                            eval_state = np.array([eval_state], dtype=np.float32)
                        elif not isinstance(eval_state, np.ndarray):
                            eval_state = np.array(eval_state, dtype=np.float32)
                        
                        # Select action (reduced exploration noise)
                        with torch.no_grad():
                            eval_action = agent.select_action(eval_state)
                            
                        # Take step using only our action (built-in opponent is used by default)
                        eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_state = eval_next_state
                    
                    eval_rewards.append(eval_episode_reward)
                
                # Store and report evaluation results
                eval_mean = np.mean(eval_rewards)
                metrics['vs_baseline_rewards'].append(eval_mean)
                print(f"\nEvaluation vs baseline at episode {episode + 1}: Mean reward = {eval_mean:.2f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                if best_params is not None:
                    agent.mean = best_params.clone()
                agent.save(save_path)
                print(f"\nModel saved to {save_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if best_params is not None:
            agent.mean = best_params.clone()
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"\nTraining interrupted due to error: {str(e)}")
        if best_params is not None:
            agent.mean = best_params.clone()
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        raise e
    
    finally:
        pbar.close()
        plt.ioff()
        metrics['training_duration'] = time.time() - start_time
        
        # Load best parameters before returning
        if best_params is not None:
            agent.mean = best_params.clone()
        
        env.close()
        
    return metrics