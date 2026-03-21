import torch

from mario_rl.losses import quantile_huber_loss


def test_quantile_huber_loss_zero_when_equal() -> None:
    b, n = 4, 8
    x = torch.zeros((b, n))
    taus = (torch.arange(n, dtype=torch.float32) + 0.5) / float(n)
    loss = quantile_huber_loss(x, x.clone(), taus=taus, kappa=1.0)
    assert loss.item() == 0.0


def test_quantile_huber_loss_positive_when_different() -> None:
    b, n = 2, 16
    current = torch.zeros((b, n))
    target = torch.ones((b, n))
    taus = (torch.arange(n, dtype=torch.float32) + 0.5) / float(n)
    loss = quantile_huber_loss(current, target, taus=taus, kappa=1.0)
    assert loss.item() > 0.0
