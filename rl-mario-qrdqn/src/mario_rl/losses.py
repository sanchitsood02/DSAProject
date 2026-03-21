from __future__ import annotations

import torch


def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    if current_quantiles.ndim != 2 or target_quantiles.ndim != 2:
        raise ValueError("expected [B, N] tensors for current_quantiles and target_quantiles")
    if current_quantiles.shape != target_quantiles.shape:
        raise ValueError("current_quantiles and target_quantiles must have same shape")
    if taus.ndim != 1 or taus.shape[0] != current_quantiles.shape[1]:
        raise ValueError("taus must be [N]")
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    delta = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
    abs_delta = delta.abs()

    huber = torch.where(abs_delta <= kappa, 0.5 * delta.pow(2), kappa * (abs_delta - 0.5 * kappa))
    indicator = (delta.detach() < 0).float()
    taus = taus.view(1, -1, 1)
    weight = (taus - indicator).abs()

    loss = (weight * huber).mean() / kappa
    return loss
