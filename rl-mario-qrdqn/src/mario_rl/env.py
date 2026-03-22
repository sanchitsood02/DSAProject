from __future__ import annotations

import importlib
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _require(pkg: str) -> None:
    raise ModuleNotFoundError(
        f"Missing optional dependency: {pkg}. Install with: pip install -e '.[mario]'"
    )


def make_mario_env(
    env_id: str,
    frame_skip: int,
    frame_stack: int,
    resize_hw: tuple[int, int],
    grayscale: bool,
    freeze_limit: int = 60,
    reward_shaping: bool = False,
    render_mode: str | None = None,
) -> Any:
    try:
        importlib.import_module("gymnasium")
    except ModuleNotFoundError:
        _require("gymnasium")

    try:
        import gym_super_mario_bros
    except ModuleNotFoundError:
        _require("gym-super-mario-bros")

    try:
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    except ModuleNotFoundError:
        _require("gym-super-mario-bros")

    try:
        from nes_py.wrappers import JoypadSpace
    except ModuleNotFoundError:
        _require("nes-py")

    kwargs: dict[str, Any] = {
        "apply_api_compatibility": True,
        "disable_env_checker": True,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym_super_mario_bros.make(env_id, **kwargs)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = RewardAndFreezeEnv(env, freeze_limit=freeze_limit)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = PreprocessObs(env, resize_hw=resize_hw, grayscale=grayscale)
    env = FrameStack(env, k=frame_stack)
    return env


class RewardAndFreezeEnv:
    def __init__(self, env: Any, freeze_limit: int) -> None:
        self.env = env
        self.freeze_limit = int(freeze_limit)
        if self.freeze_limit <= 0:
            raise ValueError("freeze_limit must be positive")

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_x: int | None = None
        self._stuck_steps = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._last_x = None
        self._stuck_steps = 0
        seed = kwargs.pop("seed", None)
        try:
            if seed is None:
                result = self.env.reset(**kwargs)
            else:
                result = self.env.reset(seed=seed, **kwargs)
        except TypeError:
            result = self.env.reset(**kwargs)
            if seed is not None and hasattr(self.env, "seed"):
                try:
                    self.env.seed(seed)
                except Exception:
                    pass

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_x = int(info.get("x_pos", 0)) if isinstance(info, dict) else 0
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            raise TypeError("unexpected step() return")

        x_pos = int(info.get("x_pos", 0))
        last_x = self._last_x if self._last_x is not None else x_pos
        dx = x_pos - last_x
        self._last_x = x_pos

        shaped = float(reward)
        shaped += float(info.get("coins", 0.0)) * 10.0
        shaped += float(info.get("x_pos", 0.0)) * 0.1
        shaped -= float(info.get("time", 0.0)) * 0.01

        if bool(info.get("flag_get", False)):
            shaped += 500.0

        if dx > 0:
            self._stuck_steps = 0
        else:
            self._stuck_steps += 1

        if self._stuck_steps >= self.freeze_limit and not (terminated or truncated):
            truncated = True
            info = dict(info)
            info["frozen"] = True
            info["anti_freeze_triggered"] = True

        return obs, shaped, bool(terminated), bool(truncated), info

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()


class MaxAndSkipEnv:
    def __init__(self, env: Any, skip: int) -> None:
        self.env = env
        self.skip = int(skip)
        if self.skip <= 0:
            raise ValueError("skip must be positive")

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._obs_buf: deque[NDArray[np.uint8]] = deque(maxlen=2)

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        self._obs_buf.clear()
        seed = kwargs.pop("seed", None)
        try:
            if seed is None:
                result = self.env.reset(**kwargs)
            else:
                result = self.env.reset(seed=seed, **kwargs)
        except TypeError:
            result = self.env.reset(**kwargs)
            if seed is not None and hasattr(self.env, "seed"):
                try:
                    self.env.seed(seed)
                except Exception:
                    pass

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        obs_arr = np.asarray(obs, dtype=np.uint8)
        self._obs_buf.append(obs_arr)
        return obs_arr, info

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        total_reward: float = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_arr = np.asarray(obs, dtype=np.uint8)
            self._obs_buf.append(obs_arr)
            total_reward += float(reward)
            if terminated or truncated:
                break

        if len(self._obs_buf) == 2:
            max_frame = np.maximum(self._obs_buf[0], self._obs_buf[-1])
        else:
            max_frame = obs_arr
        return max_frame, total_reward, terminated, truncated, info

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()


class PreprocessObs:
    def __init__(self, env: Any, resize_hw: tuple[int, int], grayscale: bool) -> None:
        import gymnasium as gym

        self.env = env
        self.resize_hw = resize_hw
        self.grayscale = bool(grayscale)
        self.action_space = env.action_space

        h, w = resize_hw
        c = 1 if self.grayscale else 3
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._transform(obs), info

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info

    def _transform(self, obs: NDArray[np.uint8]) -> NDArray[np.uint8]:
        try:
            import cv2
        except ModuleNotFoundError:
            _require("opencv-python")

        frame = np.asarray(obs, dtype=np.uint8)
        if self.grayscale:
            frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        frame = np.asarray(
            cv2.resize(
                frame,
                (self.resize_hw[1], self.resize_hw[0]),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
        if self.grayscale:
            frame = frame[None, :, :]
        else:
            frame = np.transpose(frame, (2, 0, 1))
        return frame.astype(np.uint8, copy=False)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()


class FrameStack:
    def __init__(self, env: Any, k: int) -> None:
        import gymnasium as gym

        self.env = env
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be positive")

        self.action_space = env.action_space
        obs_space = env.observation_space
        if obs_space.dtype != np.uint8:
            raise ValueError("expected uint8 observations before stacking")

        c, h, w = obs_space.shape
        self._frames: deque[NDArray[np.uint8]] = deque(maxlen=self.k)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c * self.k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(np.asarray(obs, dtype=np.uint8))
        return self._get_obs(), info

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=np.uint8))
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> NDArray[np.uint8]:
        return np.concatenate(list(self._frames), axis=0)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()
