from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QrDqnOutput:
    quantiles: torch.Tensor

    def q_values(self) -> torch.Tensor:
        return self.quantiles.mean(dim=-1)


@dataclass(frozen=True)
class DqnOutput:
    q_values: torch.Tensor


class QuantileCnn(nn.Module):
    def __init__(self, in_channels: int, n_actions: int, n_quantiles: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")
        if n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive")

        self._n_actions = int(n_actions)
        self._n_quantiles = int(n_quantiles)

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, 84, 84), dtype=torch.float32)
            enc_dim = int(self._encoder(dummy).view(1, -1).shape[1])

        self._head = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self._n_actions * self._n_quantiles),
        )

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    def forward(self, obs: torch.Tensor) -> QrDqnOutput:
        if obs.ndim != 4:
            raise ValueError("expected obs with shape [B, C, H, W]")
        x = self._encoder(obs)
        x = x.view(x.shape[0], -1)
        x = self._head(x)
        quantiles = x.view(x.shape[0], self._n_actions, self._n_quantiles)
        return QrDqnOutput(quantiles=quantiles)


class DqnCnn(nn.Module):
    def __init__(self, in_channels: int, n_actions: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")

        self._n_actions = int(n_actions)

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, 84, 84), dtype=torch.float32)
            enc_dim = int(self._encoder(dummy).view(1, -1).shape[1])

        self._head = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self._n_actions),
        )

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def forward(self, obs: torch.Tensor) -> DqnOutput:
        if obs.ndim != 4:
            raise ValueError("expected obs with shape [B, C, H, W]")
        x = self._encoder(obs)
        x = x.view(x.shape[0], -1)
        q = self._head(x)
        return DqnOutput(q_values=q)
