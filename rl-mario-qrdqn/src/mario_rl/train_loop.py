from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from mario_rl.agent import DqnAgent, QrDqnAgent
from mario_rl.config import EnvConfig, EpsilonSchedule, TrainingConfig
from mario_rl.env import make_mario_env
from mario_rl.networks import DqnCnn, QuantileCnn
from mario_rl.replay import ReplayBuffer


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _save_checkpoint(
    path: Path,
    agent: QrDqnAgent,
    step: int,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "online_state_dict": agent.online.state_dict(),
        "target_state_dict": agent.target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "extra": extra,
    }
    torch.save(payload, path)


def train(cfg: TrainingConfig, env_cfg: EnvConfig, device: torch.device) -> Path:
    _setup_logging()
    log = logging.getLogger("mario_rl.train")
    log.info("device=%s", device)

    env = make_mario_env(
        env_id=env_cfg.env_id,
        frame_skip=env_cfg.frame_skip,
        frame_stack=env_cfg.frame_stack,
        resize_hw=env_cfg.resize_hw,
        grayscale=env_cfg.grayscale,
    )

    obs_space = env.observation_space
    act_space = env.action_space
    in_channels = int(obs_space.shape[0])
    n_actions = int(act_space.n)

    online = QuantileCnn(in_channels=in_channels, n_actions=n_actions, n_quantiles=cfg.n_quantiles)
    target = QuantileCnn(in_channels=in_channels, n_actions=n_actions, n_quantiles=cfg.n_quantiles)

    agent = QrDqnAgent(
        online_net=online,
        target_net=target,
        lr=cfg.lr,
        gamma=cfg.gamma,
        eps=cfg.eps,
        huber_kappa=cfg.huber_kappa,
        grad_clip_norm=cfg.grad_clip_norm,
        device=device,
    )

    buffer = ReplayBuffer(capacity=cfg.replay_capacity)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_len = 0
    episodes = 0

    run_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "env": asdict(env_cfg)}, indent=2, default=str),
        encoding="utf-8",
    )

    last_log_step = 0
    start_time = time.time()

    for step in range(1, cfg.total_steps + 1):
        action = agent.act(obs, step=step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        buffer.add(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=done)

        episode_return += float(reward)
        episode_len += 1
        obs = next_obs

        if done:
            episodes += 1
            log.info(
                "episode=%d step=%d return=%.1f len=%d flag=%s",
                episodes,
                step,
                episode_return,
                episode_len,
                info.get("flag_get", None),
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        if buffer.size >= cfg.learning_starts and step % cfg.train_every == 0:
            batch = buffer.sample(batch_size=cfg.batch_size, device=device)
            result = agent.train_step(batch)

            if step % cfg.target_update_every == 0:
                agent.sync_target()

            if step - last_log_step >= cfg.log_every:
                last_log_step = step
                elapsed = max(time.time() - start_time, 1e-9)
                sps = float(step) / elapsed
                log.info(
                    "step=%d loss=%.5f mean_q=%.3f eps=%.3f buffer=%d sps=%.0f",
                    step,
                    result.loss,
                    result.mean_q,
                    agent.epsilon(step),
                    buffer.size,
                    sps,
                )

        if step % cfg.checkpoint_every == 0 and buffer.size >= cfg.learning_starts:
            ckpt_path = cfg.checkpoint_dir / f"qrdqn_step_{step}.pt"
            _save_checkpoint(
                ckpt_path,
                agent=agent,
                step=step,
                extra={
                    "episodes": episodes,
                    "model": {
                        "in_channels": in_channels,
                        "n_actions": n_actions,
                        "n_quantiles": cfg.n_quantiles,
                    },
                },
            )
            log.info("checkpoint=%s", ckpt_path.as_posix())

    env.close()

    final_ckpt = cfg.checkpoint_dir / "qrdqn_final.pt"
    _save_checkpoint(
        final_ckpt,
        agent=agent,
        step=cfg.total_steps,
        extra={
            "episodes": episodes,
            "model": {
                "in_channels": in_channels,
                "n_actions": n_actions,
                "n_quantiles": cfg.n_quantiles,
            },
        },
    )
    log.info("final_checkpoint=%s", final_ckpt.as_posix())
    return final_ckpt


def train_mario_new(
    env_cfg: EnvConfig,
    device: torch.device,
    episodes: int = 100,
    lr: float = 1e-4,
    batch_size: int = 128,
    gamma: float = 0.99,
    replay_capacity: int = 50_000,
    learning_starts: int = 10_000,
    target_update_every: int = 2_000,
    eps: EpsilonSchedule | None = None,
    freeze_limit: int = 60,
    model_path: Path = Path("models") / "mario_dqn.pt",
    render: bool = True,
    seed: int = 42,
) -> Path:
    _setup_logging()
    log = logging.getLogger("mario_rl.mario_new")
    log.info("device=%s", device)

    env = make_mario_env(
        env_id=env_cfg.env_id,
        frame_skip=env_cfg.frame_skip,
        frame_stack=env_cfg.frame_stack,
        resize_hw=env_cfg.resize_hw,
        grayscale=env_cfg.grayscale,
        render_mode="human" if render else None,
    )

    obs_space = env.observation_space
    act_space = env.action_space
    in_channels = int(obs_space.shape[0])
    n_actions = int(act_space.n)

    online = DqnCnn(in_channels=in_channels, n_actions=n_actions)
    target = DqnCnn(in_channels=in_channels, n_actions=n_actions)

    eps_sched = eps if eps is not None else EpsilonSchedule(start=1.0, end=0.1, decay_steps=200_000)
    agent = DqnAgent(
        online_net=online,
        target_net=target,
        lr=lr,
        gamma=gamma,
        eps=eps_sched,
        grad_clip_norm=10.0,
        device=device,
    )

    buffer = ReplayBuffer(capacity=replay_capacity)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    start_episode = 1
    step = 0

    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "online_state_dict" in ckpt:
            online_sd = cast(Mapping[str, Any], ckpt["online_state_dict"])
            target_sd = cast(
                Mapping[str, Any],
                ckpt.get("target_state_dict") or ckpt["online_state_dict"],
            )
            agent.online.load_state_dict(online_sd)
            agent.target.load_state_dict(target_sd)
            if "optimizer_state_dict" in ckpt:
                agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_episode = int(ckpt.get("episode", 0)) + 1
            step = int(ckpt.get("step", 0))
            log.info(
                "loaded_model=%s episode=%d step=%d",
                model_path.as_posix(),
                start_episode,
                step,
            )

    for ep in range(start_episode, start_episode + int(episodes)):
        obs, info = env.reset(seed=seed + ep)
        if isinstance(info, dict) and "frozen" in info:
            info.pop("frozen", None)
        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            step += 1
            action = agent.act(obs, step=step)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            buffer.add(obs=obs, action=action, reward=float(reward), next_obs=next_obs, done=done)
            obs = next_obs
            ep_return += float(reward)
            ep_len += 1

            if render:
                env.render()

            if buffer.size >= max(batch_size, learning_starts):
                batch = buffer.sample(batch_size=batch_size, device=device)
                result = agent.train_step(batch)
                if step % target_update_every == 0:
                    agent.sync_target()
                if step % 500 == 0:
                    log.info(
                        "episode=%d step=%d loss=%.5f mean_q=%.3f eps=%.3f",
                        ep,
                        step,
                        result.loss,
                        result.mean_q,
                        agent.epsilon(step),
                    )

        frozen = bool(info.get("frozen", False)) if isinstance(info, dict) else False
        x_pos = int(info.get("x_pos", -1)) if isinstance(info, dict) else -1
        log.info(
            "episode=%d shaped_return=%.1f len=%d x_pos=%d frozen=%s",
            ep,
            ep_return,
            ep_len,
            x_pos,
            frozen,
        )

        payload = {
            "episode": int(ep),
            "step": int(step),
            "online_state_dict": agent.online.state_dict(),
            "target_state_dict": agent.target.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "extra": {
                "model": {"in_channels": in_channels, "n_actions": n_actions},
                "freeze_limit": int(freeze_limit),
            },
        }
        torch.save(payload, model_path)

    env.close()
    log.info("final_model=%s", model_path.as_posix())
    return model_path
