# Volleyball Agent with Cross Entropy Method

This project implements a Cross Entropy Method (CEM) agent that learns to play volleyball in a custom gym environment. The implementation is based on the SlimeVolley environment but optimized for our specific use case.

## Features

- Custom volleyball environment with realistic physics
- Cross Entropy Method implementation for efficient learning
- Real-time visualization of training progress
- Model saving and loading capabilities
- Comprehensive training metrics visualization
- Video generation of agent gameplay

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
volleyball_agent/
├── env/
│   ├── __init__.py
│   └── volleyball_env.py
├── models/
│   ├── __init__.py
│   └── cem_model.py
├── training/
│   ├── __init__.py
│   └── train_cem.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── train_volleyball.ipynb
└── requirements.txt
```

## Usage

1. Open the Jupyter notebook `train_volleyball.ipynb`
2. Configure the hyperparameters in the notebook:
   - Environment parameters (physics, game settings)
   - CEM parameters (network architecture, population size, etc.)
   - Training parameters (number of episodes, batch size, etc.)
3. Run the training cells to start training the agent
4. Visualize the training progress and results
5. Save the trained model for later use

## Hyperparameters

### Environment Parameters
- `REF_W`, `REF_H`: Reference width and height of the game area
- `REF_U`: Ground height
- `REF_WALL_WIDTH`, `REF_WALL_HEIGHT`: Net dimensions
- `PLAYER_SPEED_X`, `PLAYER_SPEED_Y`: Player movement speeds
- `MAX_BALL_SPEED`: Maximum ball speed
- `GRAVITY`: Gravity constant
- `MAXLIVES`: Maximum number of lives per game

### CEM Parameters
- `hidden_dim`: Size of hidden layers in the neural network
- `elite_ratio`: Proportion of elite population for selection
- `population_size`: Size of the population for each generation
- `noise_std`: Standard deviation of noise for exploration
- `device`: Device to run the model on (cuda/cpu)

### Training Parameters
- `num_episodes`: Total number of training episodes
- `batch_size`: Number of episodes to collect before updating
- `eval_interval`: How often to evaluate the agent
- `save_interval`: How often to save the model
- `save_path`: Path to save the trained model

## Training Process

1. The agent starts with random actions
2. For each episode:
   - Collect experience using the current policy
   - Update the policy using CEM
   - Evaluate the agent periodically
   - Save the best model
3. Visualize training metrics and agent performance

## Visualization

The training process includes:
- Real-time plotting of episode rewards
- Moving average of population rewards
- Elite population performance tracking
- Video generation of agent gameplay

## Model Saving and Loading

Models are saved:
- When a new best reward is achieved
- At regular intervals during training
- At the end of training

To load a saved model:
```python
agent.load('path/to/saved/model.pt')
```

## Contributing

Feel free to submit issues and enhancement requests! 