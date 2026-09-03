"""
Deep Q-Learning components for the manual Warehouse Robot environment.

This file contains:
    - DQN neural network
    - Replay buffer
    - epsilon-greedy action selection
    - one optimization step using the Bellman target
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(
    state: np.ndarray,
    policy_net: DQN,
    action_dim: int,
    epsilon: float,
    device: torch.device,
) -> int:
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(action_dim)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    """One DQN update using a mini-batch from replay memory."""
    if len(replay_buffer) < batch_size:
        return None

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    action_batch = torch.as_tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_state_batch = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    done_batch = torch.as_tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Q(s, a) predicted by the online network.
    current_q = policy_net(state_batch).gather(1, action_batch)

    # Bellman target: r + gamma * max_a' Q_target(s', a')
    with torch.no_grad():
        next_q = target_net(next_state_batch).max(dim=1, keepdim=True)[0]
        target_q = reward_batch + gamma * next_q * (1.0 - done_batch)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def hard_update(target_net: DQN, policy_net: DQN) -> None:
    target_net.load_state_dict(policy_net.state_dict())


def soft_update(target_net: DQN, policy_net: DQN, tau: float = 0.01) -> None:
    """Soft update: target = tau * policy + (1 - tau) * target."""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
