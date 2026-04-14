# RL-on-CartPole

Solution to the CartPole balancing problem in the OpenAI Gym environment using Reinforcement Learning (RL) and Deep Q-Networks (DQNs). Inspired by [Deep Lizard's](https://deeplizard.com/) introduction to RL.

## Architecture

```
├── run.py                              # CLI entry point
├── DQN on CartPole.ipynb               # Original notebook (exploratory)
└── src/
    ├── config.py                       # Hyperparameters (dataclass)
    ├── model/
    │   ├── dqn.py                      # DQN neural network
    │   └── q_values.py                 # Q-value computation (current & next)
    ├── agent/
    │   ├── agent.py                    # RL agent with action selection
    │   └── strategy.py                 # Epsilon-greedy exploration strategy
    ├── environment/
    │   ├── cartpole_manager.py         # Gym environment wrapper
    │   └── screen_processor.py         # Screen cropping & transformation
    ├── training/
    │   ├── experience.py               # Experience namedtuple & replay memory
    │   ├── tensor_utils.py             # Batch tensor extraction
    │   └── trainer.py                  # Training loop orchestrator
    └── utils/
        └── plotting.py                 # Training progress visualization
```

## How It Works

1. **Environment** — CartPole-v0 from OpenAI Gym, with screen-based state representation (pixel difference between frames)
2. **DQN** — Fully connected network that maps flattened screen images to Q-values for each action (left/right)
3. **Exploration** — Epsilon-greedy strategy with exponential decay from full exploration to exploitation
4. **Experience Replay** — Stores transitions in a fixed-size buffer and samples random mini-batches for training
5. **Target Network** — Separate network updated periodically for stable Q-value targets

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run training from the command line:
```bash
python run.py --episodes 1000 --lr 0.001
```

Or explore interactively via the Jupyter notebook:
```bash
jupyter notebook "DQN on CartPole.ipynb"
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 256 | Mini-batch size for experience replay |
| `gamma` | 0.999 | Discount factor |
| `eps_start` | 1.0 | Initial exploration rate |
| `eps_end` | 0.01 | Final exploration rate |
| `eps_decay` | 0.001 | Exploration decay rate |
| `target_update` | 10 | Episodes between target network updates |
| `memory_size` | 100,000 | Replay memory capacity |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `num_episodes` | 1,000 | Total training episodes |

## Results

Training produces a live plot showing episode durations and 100-episode moving average. A trained agent typically achieves the maximum duration (200 steps) consistently after ~500 episodes.
