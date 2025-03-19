"""
Cross Entropy Method (CEM) implementation for Volleyball Agent
"""

import numpy as np
import torch
from typing import List, Dict

def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class CEMAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_size: int = 100,  # Increased from 50
        elite_ratio: float = 0.2,    # Increased from 0.1
        noise_std: float = 0.3,      # Increased for better exploration
        learning_rate: float = 0.02, # Increased from 0.01
        device: str = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device(device) if device else get_device()
        print(f"Using device: {self.device}")
        
        # Initialize parameters
        self.elite_size = int(population_size * elite_ratio)
        self.param_dim = state_dim * action_dim  # Direct mapping from state to action
        self.mean = torch.zeros(self.param_dim, device=self.device)
        self.std = torch.ones(self.param_dim, device=self.device)
        
        # Initialize population
        self.population = torch.randn(population_size, self.param_dim, device=self.device)
        
        # Keep track of best parameters
        self.best_mean = None
        self.best_reward = float('-inf')
        
    def select_action(self, state) -> np.ndarray:
        """Select action using the current mean parameters"""
        # Handle different input types
        if isinstance(state, (int, float)):
            state = np.array([state], dtype=np.float32)
        elif isinstance(state, np.ndarray):
            if state.dtype != np.float32:
                state = state.astype(np.float32)
        elif isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif not isinstance(state, torch.Tensor):
            raise TypeError(f"Unsupported state type: {type(state)}")
        
        # Convert to tensor and move to device
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = state.to(self.device)
        
        # Ensure state has correct shape
        if state.dim() == 0:
            state = state.unsqueeze(0)
        
        # Get action from mean parameters
        params = self.mean.reshape(self.state_dim, self.action_dim)
        action = torch.matmul(state, params).squeeze()
        
        # Add noise for exploration
        noise = torch.randn_like(action) * self.noise_std
        action = action + noise
        
        # Clip actions to [-1, 1] range
        action = torch.clamp(action, -1, 1)
        
        return action.cpu().numpy()
    
    def update(self, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float], elite_frac: float = None) -> dict:
        """Update parameters using CEM"""
        # Use provided elite_frac if given, otherwise use instance value
        elite_ratio = elite_frac if elite_frac is not None else self.elite_ratio
        elite_size = int(self.population_size * elite_ratio)
        
        # Convert states to numpy array and ensure correct type
        if isinstance(states, list):
            # Handle mixed types in states list
            processed_states = []
            for s in states:
                if isinstance(s, (int, float)):
                    processed_states.append(np.array([s], dtype=np.float32))
                elif isinstance(s, np.ndarray):
                    if s.dtype != np.float32:
                        processed_states.append(s.astype(np.float32))
                    else:
                        processed_states.append(s)
                else:
                    raise TypeError(f"Unsupported state type in list: {type(s)}")
            states = np.array(processed_states)
        elif isinstance(states, np.ndarray):
            if states.dtype != np.float32:
                states = states.astype(np.float32)
        else:
            raise TypeError(f"Unsupported states type: {type(states)}")
        
        # Convert actions and rewards to numpy arrays first (avoiding the warning)
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        
        if isinstance(rewards, list):
            rewards = np.array(rewards, dtype=np.float32)
        
        # Convert inputs to tensors
        states = torch.from_numpy(states).float()
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        elif not isinstance(actions, torch.Tensor):
            raise TypeError(f"Unsupported actions type: {type(actions)}")
            
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float()
        elif not isinstance(rewards, torch.Tensor):
            raise TypeError(f"Unsupported rewards type: {type(rewards)}")
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # Ensure states have correct shape
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Store original rewards mean before normalization
        original_rewards_mean = torch.mean(rewards).item()
        original_rewards_std = torch.std(rewards).item()
        
        # Normalize rewards (only if they have non-zero variance)
        if rewards.std() > 1e-8:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()  # Center them at least
        
        # Generate population of parameters
        population_rewards = []
        for i in range(self.population_size):
            # Apply noise to parameters
            params = self.mean + self.std * self.population[i]
            params = params.reshape(self.state_dim, self.action_dim)
            
            # Evaluate parameters on batch
            predicted_actions = torch.matmul(states, params)
            action_loss = torch.mean((predicted_actions - actions) ** 2)
            
            # Weight loss by rewards
            weighted_loss = action_loss * normalized_rewards.mean()
            population_rewards.append(-weighted_loss.item())  # Negative because we want to maximize rewards
        
        # Select elite population
        elite_indices = torch.tensor(population_rewards).to(self.device).argsort()[-elite_size:]
        elite_population = self.population[elite_indices]
        
        # Update mean and std
        self.mean = self.mean + self.learning_rate * (elite_population.mean(0) - self.mean)
        self.std = self.std + self.learning_rate * (elite_population.std(0) - self.std)
        
        # Ensure std doesn't get too small
        self.std = torch.clamp(self.std, min=0.01)
        
        # Track best parameters if these rewards are better
        if original_rewards_mean > self.best_reward:
            self.best_reward = original_rewards_mean
            self.best_mean = self.mean.clone()
        
        # Update population
        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)
        
        return {
            "mean_reward": original_rewards_mean,  # Use original unnormalized mean
            "mean_reward_std": original_rewards_std,  # Return std dev too
            "elite_reward": np.mean([population_rewards[i] for i in elite_indices.cpu().numpy()])
        }
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'best_mean': self.best_mean if self.best_mean is not None else self.mean,
            'best_reward': self.best_reward,
            'device': self.device,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'population_size': self.population_size,
                'elite_ratio': self.elite_ratio,
                'noise_std': self.noise_std,
                'learning_rate': self.learning_rate
            }
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load parameters
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        
        # Load best parameters if available
        if 'best_mean' in checkpoint:
            self.best_mean = checkpoint['best_mean']
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
        
        # Load hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.state_dim = hyperparams.get('state_dim', self.state_dim)
            self.action_dim = hyperparams.get('action_dim', self.action_dim)
            self.population_size = hyperparams.get('population_size', self.population_size)
            self.elite_ratio = hyperparams.get('elite_ratio', self.elite_ratio)
            self.noise_std = hyperparams.get('noise_std', self.noise_std)
            self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
        
        # Recreate population
        self.elite_size = int(self.population_size * self.elite_ratio)
        self.param_dim = self.state_dim * self.action_dim
        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)