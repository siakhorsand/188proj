"""
Script to generate Jupyter notebook for volleyball agent training
"""

import nbformat as nbf
import json
from datetime import datetime

def create_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Title
    title = nbf.v4.new_markdown_cell("""# Slime Volleyball Agent Training
This notebook implements training for the Slime Volleyball environment using the Cross Entropy Method (CEM).""")
    nb.cells.append(title)
    
    # Imports and Setup
    imports = nbf.v4.new_code_cell("""import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from IPython.display import display, HTML, clear_output
import time
from matplotlib import animation
import gym
import os

# Monkey patch the gym registration function to handle duplicate registrations
original_register = gym.envs.registration.register

def patched_register(id, **kwargs):
    if id in gym.envs.registry.env_specs:
        return
    return original_register(id, **kwargs)

# Apply the patch
gym.envs.registration.register = patched_register

# Now import slimevolleygym
import slimevolleygym

# Create environment
env = gym.make('SlimeVolley-v0')

# Import our training modules
from models.cem_model import CEMAgent
from training.train_cem import train_cem""")
    nb.cells.append(imports)
    
    # Device Setup
    device_setup = nbf.v4.new_code_cell("""# Set up device for PyTorch
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

device = get_device()""")
    nb.cells.append(device_setup)
    
    # Environment Info
    env_info = nbf.v4.new_markdown_cell("""## Environment Information
The Slime Volleyball environment provides:
- Observation space: 12-dimensional state vector containing:
  - Agent position (x, y) and velocity (vx, vy)
  - Ball position (x, y) and velocity (vx, vy)
  - Opponent position (x, y) and velocity (vx, vy)
- Action space: 3 binary actions [forward, backward, jump]
- Reward: +1 for scoring, -1 for being scored on
- Episode ends when one player scores""")
    nb.cells.append(env_info)
    
    # Environment Setup
    env_setup = nbf.v4.new_code_cell("""# Print environment details
print("Environment Information:")
print(f"Observation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# Initialize the CEM agent with correct parameters
agent = CEMAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=3,  # [forward, backward, jump]
    population_size=50,  # Size of the population for CEM
    elite_ratio=0.2,    # Top 20% are elite
    noise_std=0.1,      # Standard deviation of noise
    learning_rate=0.01  # Learning rate for parameter updates
)

print("\\nAgent Architecture:")
print(f"Input dimensions: {env.observation_space.shape[0]}")
print(f"Output dimensions: {env.action_space.n}")
print(f"Population size: {agent.population_size}")
print(f"Elite ratio: {agent.elite_ratio}")""")
    nb.cells.append(env_setup)
    
    # Test Environment
    test_env = nbf.v4.new_markdown_cell("""## Environment Test
Let's test the environment by running a few random actions and displaying the results.""")
    nb.cells.append(test_env)
    
    test_code = nbf.v4.new_code_cell("""def test_environment(env, num_steps=300):  # 10 seconds at 30 FPS
    frames = []
    total_reward = 0
    state = env.reset()
    
    print("Running environment test...")
    for step in range(num_steps):
        # Random action: one hot vector
        action = np.zeros(3)
        action[np.random.randint(3)] = 1  # Set one action to 1
        
        # Step environment
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Render and capture frame
        try:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        except Exception as e:
            print(f"Warning: Could not render frame: {e}")
            continue
        
        if done:
            print(f"Episode finished after {step} steps with reward {total_reward}")
            state = env.reset()
            total_reward = 0
    
    env.close()
    return frames

# Run test and create animation
try:
    frames = test_environment(env)
    
    if frames:
        # Create animation
        fig = plt.figure(figsize=(10, 6))
        patch = plt.imshow(frames[0])
        plt.axis('off')
        plt.title('Slime Volleyball Environment Test')

        def animate(i):
            patch.set_array(frames[i])
            return [patch]

        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames),
            interval=1000/30,  # 30 FPS
            blit=True
        )

        plt.close()
        display(HTML(anim.to_jshtml()))
    else:
        print("No frames were captured. Please check if XQuartz is running and properly configured.")
except Exception as e:
    print(f"Error during environment test: {e}")
    print("Please make sure XQuartz is running and properly configured.")""")
    nb.cells.append(test_code)
    
    # Training Configuration
    train_config = nbf.v4.new_markdown_cell("""## Training Configuration
Set up the hyperparameters for training the agent using CEM.""")
    nb.cells.append(train_config)
    
    config_code = nbf.v4.new_code_cell("""# Training hyperparameters
hyperparameters = {
    'num_episodes': 100,      # Total episodes to train
    'batch_size': 16,         # Episodes per batch
    'elite_frac': 0.2,        # Top fraction of episodes to use for update
    'eval_interval': 10,      # Episodes between evaluations
    'save_interval': 20,      # Episodes between model saves
}

print("Training Configuration:")
for key, value in hyperparameters.items():
    print(f"{key}: {value}")""")
    nb.cells.append(config_code)
    
    # Training Loop
    train_desc = nbf.v4.new_markdown_cell("""## Training Loop
Train the agent using the Cross Entropy Method:
1. Run multiple episodes with the current policy
2. Select the top performing episodes
3. Update the policy using the elite episodes
4. Repeat""")
    nb.cells.append(train_desc)
    
    train_code = nbf.v4.new_code_cell("""# Create save directory if it doesn't exist
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Training loop
print("Starting training...")
metrics = train_cem(
    env=env,
    agent=agent,
    device=device,
    num_episodes=hyperparameters['num_episodes'],
    batch_size=hyperparameters['batch_size'],
    elite_frac=hyperparameters['elite_frac'],
    eval_interval=hyperparameters['eval_interval'],
    save_interval=hyperparameters['save_interval'],
    save_dir=save_dir
)
print("Training complete!")""")
    nb.cells.append(train_code)
    
    # Visualization
    viz = nbf.v4.new_markdown_cell("""## Training Visualization
Plot the training progress and metrics.""")
    nb.cells.append(viz)
    
    viz_code = nbf.v4.new_code_cell("""# Plot training metrics
plt.figure(figsize=(15, 5))

# Plot episode rewards
plt.subplot(1, 2, 1)
plt.plot(metrics['episode_rewards'], 'b-', alpha=0.3, label='Episode Reward')
plt.plot(metrics['mean_rewards'], 'r-', label='Moving Average')
plt.fill_between(
    range(len(metrics['mean_rewards'])),
    np.array(metrics['mean_rewards']) - np.array(metrics['std_rewards']),
    np.array(metrics['mean_rewards']) + np.array(metrics['std_rewards']),
    alpha=0.2, color='r'
)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.grid(True)
plt.legend()

# Plot win rate
plt.subplot(1, 2, 2)
window = 10
wins = [r > 0 for r in metrics['episode_rewards']]
win_rate = [sum(wins[max(0, i-window):i])/min(i, window) 
            for i in range(1, len(wins)+1)]
plt.plot(win_rate, 'g-', label='Win Rate')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title(f'Win Rate (Moving Average, Window={window})')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()""")
    nb.cells.append(viz_code)
    
    # Save Results
    save = nbf.v4.new_markdown_cell("""## Save Results
Save the training metrics and configuration for later analysis.""")
    nb.cells.append(save)
    
    save_code = nbf.v4.new_code_cell("""# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    'metrics': metrics,
    'hyperparameters': hyperparameters,
    'final_mean_reward': float(np.mean(metrics['episode_rewards'][-10:])),
    'best_reward': float(max(metrics['episode_rewards'])),
    'final_win_rate': win_rate[-1],
    'training_duration': metrics.get('training_duration', 0)
}

# Save to file
results_file = f'results/training_results_{timestamp}.json'
os.makedirs('results', exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {results_file}")

# Print summary
print("\\nTraining Summary:")
print(f"Final average reward (last 10 episodes): {results['final_mean_reward']:.2f}")
print(f"Best episode reward: {results['best_reward']:.2f}")
print(f"Final win rate: {results['final_win_rate']:.2%}")
print(f"Training duration: {results['training_duration']:.2f} seconds")""")
    nb.cells.append(save_code)
    
    # Write the notebook
    with open('train_volleyball.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook() 