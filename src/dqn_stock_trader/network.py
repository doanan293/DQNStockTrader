"""Conv1D Q-network kept from the original project's architecture."""

import torch
from torch import nn
from typing_extensions import override


class QNetwork(nn.Module):
    """Conv1D(64,k6) -> Dropout -> Conv1D(32,k3) -> Conv1D(16,k2) -> MaxPool -> Dense(32) -> Q."""

    def __init__(self, state_size: int, n_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, state_size)).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, 32), nn.ReLU(), nn.Linear(32, n_actions))

    @override
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(states.unsqueeze(1)))
