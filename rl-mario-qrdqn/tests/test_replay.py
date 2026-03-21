import numpy as np
import torch

from mario_rl.replay import ReplayBuffer


def test_replay_add_and_sample_shapes() -> None:
    buf = ReplayBuffer(capacity=100)
    obs = np.zeros((4, 84, 84), dtype=np.uint8)

    for i in range(50):
        buf.add(obs=obs, action=i % 6, reward=1.0, next_obs=obs, done=False)

    batch = buf.sample(batch_size=32, device=torch.device("cpu"))
    assert batch.obs.shape == (32, 4, 84, 84)
    assert batch.next_obs.shape == (32, 4, 84, 84)
    assert batch.actions.shape == (32,)
    assert batch.rewards.shape == (32,)
    assert batch.dones.shape == (32,)
    assert batch.obs.dtype == torch.float32
