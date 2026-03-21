from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from mario_rl.config import EnvConfig
from mario_rl.env import make_mario_env
from mario_rl.networks import QuantileCnn


def evaluate(
    checkpoint_path: Path,
    env_cfg: EnvConfig,
    device: torch.device,
    episodes: int,
    record_video: bool,
    video_dir: Path,
) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("mario_rl.eval")

    env = make_mario_env(
        env_id=env_cfg.env_id,
        frame_skip=env_cfg.frame_skip,
        frame_stack=env_cfg.frame_stack,
        resize_hw=env_cfg.resize_hw,
        grayscale=env_cfg.grayscale,
    )

    if record_video:
        try:
            import gymnasium as gym
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "gymnasium is required for video recording. Install with: pip install -e '.[mario]'"
            ) from e

        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,
        )

    obs_space = env.observation_space
    act_space = env.action_space
    in_channels = int(obs_space.shape[0])
    n_actions = int(act_space.n)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ckpt.get("extra", {}).get("model", {})
    n_quantiles = int(model.get("n_quantiles", 200))
    online = QuantileCnn(in_channels=in_channels, n_actions=n_actions, n_quantiles=n_quantiles)
    online.load_state_dict(ckpt["online_state_dict"])
    online.to(device)
    online.eval()

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_t = torch.as_tensor(obs, device=device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)
            if obs_t.dtype == torch.uint8:
                obs_t = obs_t.float().div(255.0)

            with torch.no_grad():
                q = online(obs_t).q_values()[0]
                action = int(torch.argmax(q).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        log.info("episode=%d return=%.1f len=%d", ep + 1, ep_ret, ep_len)

    env.close()
    return {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "len_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
