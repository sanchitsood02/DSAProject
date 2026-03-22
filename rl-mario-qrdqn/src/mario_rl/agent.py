from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from mario_rl.config import EpsilonSchedule
from mario_rl.losses import quantile_huber_loss
from mario_rl.networks import DqnCnn, QuantileCnn
from mario_rl.replay import ReplayBatch


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    mean_q: float


class QrDqnAgent:
    def __init__(
        self,
        online_net: QuantileCnn,
        target_net: QuantileCnn,
        lr: float,
        gamma: float,
        eps: EpsilonSchedule,
        huber_kappa: float,
        grad_clip_norm: float,
        device: torch.device,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")

        self.device = device
        self.online = online_net.to(device)
        self.target = target_net.to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.eps = eps
        self.huber_kappa = float(huber_kappa)
        self.grad_clip_norm = float(grad_clip_norm)

        n = self.online.n_quantiles
        taus = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / float(n)
        self.taus = taus

    def epsilon(self, step: int) -> float:
        if step <= 0:
            return self.eps.start
        frac = min(float(step) / float(self.eps.decay_steps), 1.0)
        return float(self.eps.start + frac * (self.eps.end - self.eps.start))

    @torch.no_grad()
    def act(self, obs: NDArray[np.uint8], step: int) -> int:
        eps = self.epsilon(step)
        if np.random.rand() < eps:
            return int(np.random.randint(0, self.online.n_actions))

        obs_t = torch.as_tensor(obs, device=self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.dtype == torch.uint8:
            obs_t = obs_t.float().div(255.0)
        q = self.online(obs_t).q_values()[0]
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self, batch: ReplayBatch) -> TrainStepResult:
        self.online.train()
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones

        out = self.online(obs).quantiles
        qa = out.gather(1, actions.view(-1, 1, 1).expand(-1, 1, out.shape[-1])).squeeze(1)

        with torch.no_grad():
            next_out_online = self.online(next_obs).q_values()
            next_actions = torch.argmax(next_out_online, dim=1)

            next_out_target = self.target(next_obs).quantiles
            next_qa = next_out_target.gather(
                1, next_actions.view(-1, 1, 1).expand(-1, 1, next_out_target.shape[-1])
            ).squeeze(1)

            target = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma * next_qa

        loss = quantile_huber_loss(qa, target, self.taus, kappa=self.huber_kappa)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_clip_norm)

        self.optimizer.step()
        mean_q = float(qa.mean().detach().cpu().item())
        return TrainStepResult(loss=float(loss.detach().cpu().item()), mean_q=mean_q)


class DqnAgent:
    def __init__(
        self,
        online_net: DqnCnn,
        target_net: DqnCnn,
        lr: float,
        gamma: float,
        eps: EpsilonSchedule,
        grad_clip_norm: float,
        device: torch.device,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")

        self.device = device
        self.online = online_net.to(device)
        self.target = target_net.to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.eps = eps
        self.grad_clip_norm = float(grad_clip_norm)

    def epsilon(self, step: int) -> float:
        if step <= 0:
            return self.eps.start
        frac = min(float(step) / float(self.eps.decay_steps), 1.0)
        return float(self.eps.start + frac * (self.eps.end - self.eps.start))

    @torch.no_grad()
    def act(self, obs: NDArray[np.uint8], step: int) -> int:
        eps = self.epsilon(step)
        if np.random.rand() < eps:
            return int(np.random.randint(0, self.online.n_actions))

        obs_t = torch.as_tensor(obs, device=self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.dtype == torch.uint8:
            obs_t = obs_t.float().div(255.0)
        q = self.online(obs_t).q_values[0]
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def train_step(self, batch: ReplayBatch) -> TrainStepResult:
        self.online.train()
        obs = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_obs
        dones = batch.dones

        q = self.online(obs).q_values
        qa = q.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(next_obs).q_values
            next_max = next_q.max(dim=1).values
            target = rewards + (1.0 - dones) * self.gamma * next_max

        loss = F.smooth_l1_loss(qa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_clip_norm)

        self.optimizer.step()
        mean_q = float(qa.mean().detach().cpu().item())
        return TrainStepResult(loss=float(loss.detach().cpu().item()), mean_q=mean_q)
