"""
Script to test different hyperparameter combinations for the CEM agent
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
import json
from datetime import datetime

from env.volleyball_env import VolleyballEnv
from models.cem_model import CEMAgent
from training.train_cem import train_cem

def test_hyperparameters():
    # Define hyperparameter combinations to test
    param_combinations = {
        'population_size': [50, 100, 200],
        'elite_ratio': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.005, 0.01]
    }
    
    # Generate all combinations
    keys = param_combinations.keys()
    values = param_combinations.values()
    combinations = list(product(*values))
    
    # Test results storage
    results = []
    
    # Number of episodes for each test
    test_episodes = 100
    
    print(f"Testing {len(combinations)} hyperparameter combinations...")
    
    for i, params in enumerate(combinations):
        # Create parameter dictionary
        param_dict = dict(zip(keys, params))
        
        print(f"\nTesting combination {i+1}/{len(combinations)}")
        print("Parameters:", param_dict)
        
        # Create environment and agent
        env = VolleyballEnv()
        agent = CEMAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            population_size=param_dict['population_size'],
            elite_ratio=param_dict['elite_ratio'],
            learning_rate=param_dict['learning_rate']
        )
        
        # Train and collect metrics
        metrics = train_cem(
            env=env,
            agent=agent,
            num_episodes=test_episodes,
            batch_size=10,
            eval_interval=20,
            save_interval=test_episodes,
            save_path=f"models/cem_test_{i}.pt"
        )
        
        # Store results
        result = {
            'parameters': param_dict,
            'final_mean_reward': np.mean(metrics['episode_rewards'][-10:]),
            'best_reward': max(metrics['episode_rewards']),
            'std_reward': np.std(metrics['episode_rewards'][-10:]),
            'convergence_episode': np.argmax(metrics['episode_rewards']) + 1
        }
        results.append(result)
        
        print(f"Results for combination {i+1}:")
        print(f"Final Mean Reward: {result['final_mean_reward']:.2f}")
        print(f"Best Reward: {result['best_reward']:.2f}")
        print(f"Std Reward: {result['std_reward']:.2f}")
        print(f"Convergence Episode: {result['convergence_episode']}")
    
    # Find best combination
    best_result = max(results, key=lambda x: x['final_mean_reward'])
    
    print("\nBest Hyperparameter Combination:")
    print("Parameters:", best_result['parameters'])
    print(f"Final Mean Reward: {best_result['final_mean_reward']:.2f}")
    print(f"Best Reward: {best_result['best_reward']:.2f}")
    print(f"Std Reward: {best_result['std_reward']:.2f}")
    print(f"Convergence Episode: {best_result['convergence_episode']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': best_result,
            'test_episodes': test_episodes
        }, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    
    return best_result

if __name__ == "__main__":
    best_params = test_hyperparameters() 