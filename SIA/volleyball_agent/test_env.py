"""
Test script for the volleyball environment
"""

import numpy as np
from env.volleyball_env import VolleyballEnv
import time

def main():
    # Create environment
    env = VolleyballEnv()
    
    # Run for 1000 steps
    state = env.reset()
    print("Initial state shape:", state.shape)
    print("Action space:", env.action_space)
    
    total_reward = 0
    steps = 0
    
    print("\nRunning environment test...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Random action
            action = np.zeros(3)
            action[np.random.randint(3)] = np.random.choice([-1, 1])
            
            # Step environment
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            env.render(mode='human')
            time.sleep(1/60)  # Cap at 60 FPS
            
            if done:
                print(f"\nEpisode finished after {steps} steps with reward {total_reward}")
                state = env.reset()
                total_reward = 0
                steps = 0
                
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main() 