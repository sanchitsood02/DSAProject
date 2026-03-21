from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._size = 0
        self._pos = 0

        self._obs: NDArray[np.uint8] | None = None
        self._next_obs: NDArray[np.uint8] | None = None
        self._actions: NDArray[np.int64] | None = None
        self._rewards: NDArray[np.float32] | None = None
        self._dones: NDArray[np.bool_] | None = None

    @property
    def size(self) -> int:
        return self._size

    def _lazy_init(self, obs: NDArray[np.uint8]) -> None:
        obs = np.asarray(obs, dtype=np.uint8)
        self._obs = np.empty((self._capacity, *obs.shape), dtype=obs.dtype)
        self._next_obs = np.empty((self._capacity, *obs.shape), dtype=obs.dtype)
        self._actions = np.empty((self._capacity,), dtype=np.int64)
        self._rewards = np.empty((self._capacity,), dtype=np.float32)
        self._dones = np.empty((self._capacity,), dtype=np.bool_)

    def add(
        self,
        obs: NDArray[np.uint8],
        action: int,
        reward: float,
        next_obs: NDArray[np.uint8],
        done: bool,
    ) -> None:
        if self._obs is None:
            self._lazy_init(obs)
        assert self._obs is not None
        assert self._next_obs is not None
        assert self._actions is not None
        assert self._rewards is not None
        assert self._dones is not None

        self._obs[self._pos] = obs
        self._next_obs[self._pos] = next_obs
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._dones[self._pos] = bool(done)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError("batch_size larger than buffer size")

        assert self._obs is not None
        assert self._next_obs is not None
        assert self._actions is not None
        assert self._rewards is not None
        assert self._dones is not None

        idx = np.random.randint(0, self._size, size=(batch_size,))
        obs = torch.as_tensor(self._obs[idx], device=device)
        next_obs = torch.as_tensor(self._next_obs[idx], device=device)
        actions = torch.as_tensor(self._actions[idx], device=device, dtype=torch.long)
        rewards = torch.as_tensor(self._rewards[idx], device=device, dtype=torch.float32)
        dones = torch.as_tensor(self._dones[idx], device=device, dtype=torch.float32)

        if obs.dtype == torch.uint8:
            obs = obs.float().div_(255.0)
            next_obs = next_obs.float().div_(255.0)

        return ReplayBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )
