from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.config import HyperParams
from src.model.dqn import DQN
from src.model.q_values import QValues
from src.agent.agent import Agent
from src.agent.strategy import EpsilonGreedyStrategy
from src.environment.cartpole_manager import CartPoleEnvManager
from src.training.experience import Experience, ReplayMemory
from src.training.tensor_utils import extract_tensors
from src.utils.plotting import plot


def train(params: HyperParams = None):
    if params is None:
        params = HyperParams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(params.eps_start, params.eps_end, params.eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(params.memory_size)

    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=params.learning_rate)

    episode_durations = []

    for episode in range(params.num_episodes):
        em.reset()
        state = em.get_state()

        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(params.batch_size):
                experiences = memory.sample(params.batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * params.gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if em.done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

        if episode % params.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    em.close()
    return episode_durations
