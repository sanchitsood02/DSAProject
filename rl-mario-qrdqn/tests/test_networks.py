import torch

from mario_rl.networks import QuantileCnn


def test_quantile_cnn_output_shapes() -> None:
    net = QuantileCnn(in_channels=4, n_actions=6, n_quantiles=200)
    obs = torch.zeros((2, 4, 84, 84), dtype=torch.float32)
    out = net(obs)
    assert out.quantiles.shape == (2, 6, 200)
    q = out.q_values()
    assert q.shape == (2, 6)
