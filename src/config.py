from dataclasses import dataclass


@dataclass
class HyperParams:
    batch_size: int = 256
    gamma: float = 0.999
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.001
    target_update: int = 10
    memory_size: int = 100000
    learning_rate: float = 0.001
    num_episodes: int = 1000
