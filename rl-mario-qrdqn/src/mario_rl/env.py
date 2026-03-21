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

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env = PreprocessObs(env, resize_hw=resize_hw, grayscale=grayscale)
    env = FrameStack(env, k=frame_stack)
    return env


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
        obs, info = self.env.reset(**kwargs)
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
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame,
            (self.resize_hw[1], self.resize_hw[0]),
            interpolation=cv2.INTER_AREA,
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
