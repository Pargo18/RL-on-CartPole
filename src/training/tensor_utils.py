import torch
from src.training.experience import Experience


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)

    return states, actions, rewards, next_states
