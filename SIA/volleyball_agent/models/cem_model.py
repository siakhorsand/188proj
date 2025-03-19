"""
Enhanced Cross Entropy Method (CEM) implementation with neural network policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Policy(nn.Module):
    """Neural network policy for CEM agent"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output between -1 and 1
        return x
    
    def get_params(self):
        """Get flattened parameters of the network"""
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params, 0)
    
    def set_params(self, flat_params):
        """Set flattened parameters of the network"""
        offset = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data.copy_(flat_params[offset:offset + param_size].view(param.size()))
            offset += param_size


class CEMAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_size: int = 100,  # Increased from 50
        elite_ratio: float = 0.2,    # Increased from 0.1
        noise_std: float = 0.2,      # Increased from 0.1
        learning_rate: float = 0.05,  # Increased from 0.01
        hidden_dim: int = 64,        # New parameter for NN size
        device: str = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Set device
        self.device = torch.device(device) if device else get_device()
        print(f"Using device: {self.device}")
        
        # Create policy network
        self.policy = Policy(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize parameters
        self.elite_size = int(population_size * elite_ratio)
        self.param_dim = sum(p.numel() for p in self.policy.parameters())
        
        # Initialize mean and std for CEM
        self.mean = torch.zeros(self.param_dim, device=self.device)
        self.std = torch.ones(self.param_dim, device=self.device)
        
        # Initialize population
        self.population = torch.randn(population_size, self.param_dim, device=self.device)
        
        # Store best parameters found so far
        self.best_params = None
        self.best_reward = float('-inf')
        
        # Apply initial parameters to policy
        self.policy.set_params(self.mean)
        
        # Noise decay
        self.noise_decay = 0.995  # Will multiply noise by this factor after each update
        
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
        
        # Get action from policy
        with torch.no_grad():
            action = self.policy(state)
            
            # Add noise for exploration during training
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
        
        # Store original rewards mean before any processing
        original_rewards_mean = float(np.mean(rewards))
        original_rewards_std = float(np.std(rewards))
        
        # Convert inputs to tensors
        states = torch.from_numpy(states).float().to(self.device)
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.device)
        elif not isinstance(actions, torch.Tensor):
            raise TypeError(f"Unsupported actions type: {type(actions)}")
        else:
            actions = actions.to(self.device)
            
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).float().to(self.device)
        elif not isinstance(rewards, torch.Tensor):
            raise TypeError(f"Unsupported rewards type: {type(rewards)}")
        else:
            rewards = rewards.to(self.device)
        
        # Ensure states have correct shape
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Normalize rewards (only if they have non-zero variance)
        if rewards.std() > 1e-8:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()  # Center them at least
        
        # Generate population of parameters
        population_rewards = []
        
        for i in range(self.population_size):
            # Apply population parameters to policy
            params = self.mean + self.std * self.population[i]
            self.policy.set_params(params)
            
            # Evaluate parameters on batch
            with torch.no_grad():
                predicted_actions = []
                for state in states:
                    predicted_action = self.policy(state.unsqueeze(0))
                    predicted_actions.append(predicted_action.squeeze(0))
                predicted_actions = torch.stack(predicted_actions)
                
                # Calculate loss (MSE)
                action_loss = F.mse_loss(predicted_actions, actions)
                
                # Weight loss by rewards
                weighted_loss = action_loss * normalized_rewards.mean()
                
            population_rewards.append(-weighted_loss.item())  # Negative because we want to maximize rewards
        
        # Select elite population
        elite_indices = torch.tensor(population_rewards).to(self.device).argsort()[-elite_size:]
        elite_population = self.population[elite_indices]
        
        # Calculate elite reward
        elite_reward = float(np.mean([population_rewards[i] for i in elite_indices.cpu().numpy()]))
        
        # Update mean and std
        old_mean = self.mean.clone()
        self.mean = old_mean + self.learning_rate * (elite_population.mean(0) - old_mean)
        self.std = torch.clamp(
            self.std + self.learning_rate * (elite_population.std(0) - self.std),
            min=0.01,
            max=1.0
        )
        
        # Decay noise for better convergence
        self.noise_std *= self.noise_decay
        
        # Update the policy with the new mean
        self.policy.set_params(self.mean)
        
        # Track best parameters
        if original_rewards_mean > self.best_reward:
            self.best_reward = original_rewards_mean
            self.best_params = self.mean.clone()
        
        # Create new population
        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)
        
        return {
            "mean_reward": original_rewards_mean,
            "mean_reward_std": original_rewards_std,
            "elite_reward": elite_reward,
            "noise_std": self.noise_std
        }
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'best_params': self.best_params,
            'best_reward': self.best_reward,
            'device': self.device,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'population_size': self.population_size,
                'elite_ratio': self.elite_ratio,
                'noise_std': self.noise_std,
                'learning_rate': self.learning_rate,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.state_dim = hyperparams['state_dim']
        self.action_dim = hyperparams['action_dim']
        self.population_size = hyperparams['population_size']
        self.elite_ratio = hyperparams['elite_ratio']
        self.noise_std = hyperparams['noise_std']
        self.learning_rate = hyperparams['learning_rate']
        self.hidden_dim = hyperparams['hidden_dim']
        
        # Recreate policy with loaded hyperparameters
        self.policy = Policy(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Load parameters
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.best_params = checkpoint['best_params']
        self.best_reward = checkpoint['best_reward']
        
        # Apply best parameters to policy
        if self.best_params is not None:
            self.policy.set_params(self.best_params)
        else:
            self.policy.set_params(self.mean)
            
        # Recreate population
        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)