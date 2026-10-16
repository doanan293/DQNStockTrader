"""DQN agent with vanilla, fixed-target and Double DQN updates."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from dqn_stock_trader.network import QNetwork
from dqn_stock_trader.replay import Batch, ReplayBuffer, Transition

Variant = Literal["dqn", "fixed_target", "double"]


@dataclass(frozen=True)
class AgentConfig:
    variant: Variant = "double"
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 10_000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_sync_steps: int = 1_000


class DQNAgent:
    def __init__(self, state_size: int, n_actions: int, config: AgentConfig, seed: int) -> None:
        torch.manual_seed(seed)
        self.config = config
        self.n_actions = n_actions
        self.online: nn.Module = QNetwork(state_size, n_actions)
        self.target: nn.Module = (
            self.online if config.variant == "dqn" else copy.deepcopy(self.online)
        )
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(config.buffer_size, seed)
        self.epsilon = config.epsilon_start
        self.learn_steps = 0
        self._rng = np.random.default_rng(seed)

    def act(self, state: npt.NDArray[np.float64], *, greedy: bool) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        self.online.eval()
        with torch.no_grad():
            q = self.online(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        self.online.train()
        return int(q.argmax(dim=1).item())

    def remember(self, transition: Transition) -> None:
        self.buffer.push(transition)

    def compute_targets(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            if self.config.variant == "double":
                best = self.online(batch.next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(batch.next_states).gather(1, best).squeeze(1)
            else:
                next_q = self.target(batch.next_states).max(dim=1).values
        return batch.rewards + self.config.gamma * next_q * (1.0 - batch.dones)

    def learn(self) -> float | None:
        if len(self.buffer) < self.config.batch_size:
            return None
        batch = self.buffer.sample(self.config.batch_size)
        targets = self.compute_targets(batch)
        q = self.online(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1
        if self.target is not self.online and self.learn_steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online.state_dict(), path)

    def load(self, path: Path) -> None:
        self.online.load_state_dict(torch.load(path, weights_only=True))
        if self.target is not self.online:
            self.target.load_state_dict(self.online.state_dict())
