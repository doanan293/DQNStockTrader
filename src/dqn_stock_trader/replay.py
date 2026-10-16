"""Fixed-size experience replay buffer."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass(frozen=True)
class Transition:
    state: npt.NDArray[np.float64]
    action: int
    reward: float
    next_state: npt.NDArray[np.float64]
    done: bool


@dataclass(frozen=True)
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> Batch:
        picks = [self._items[i] for i in self._rng.choice(len(self._items), batch_size)]
        return Batch(
            states=torch.tensor(np.stack([t.state for t in picks]), dtype=torch.float32),
            actions=torch.tensor([t.action for t in picks], dtype=torch.int64),
            rewards=torch.tensor([t.reward for t in picks], dtype=torch.float32),
            next_states=torch.tensor(np.stack([t.next_state for t in picks]), dtype=torch.float32),
            dones=torch.tensor([t.done for t in picks], dtype=torch.float32),
        )
