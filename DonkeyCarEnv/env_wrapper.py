import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from typing import Any
import os
import torch

from DonkeyCarEnv.donkey_gym.envs.vae_env import DonkeyVAEEnv
from DonkeyCarEnv.config_env import (
    LEVEL,
    FRAME_SKIP,
    MIN_THROTTLE,
    MAX_THROTTLE,
    MAX_CTE_ERROR,
    N_STACK,
    N_COMMAND_HISTORY,
)
from DonkeyCarEnv.ae.autoencoder import load_ae
import pathlib

MAX_EPISODE_STEPS = (
    1000  # fallback horizon for wrappers expecting a finite episode length
)
VAE_ARCHIVE_DIR = os.environ.get(
    "VAE_ARCHIVE_DIR",
    pathlib.Path(__file__).parent
    / "ae_pretrained_weights"
    / "ae-32-level0"
    / "vae.pkl",
)  # Fallback path for VAE weights


class GymnasiumDonkeyWrapper(gym.Env):
    """A Gymnasium wrapper for the DonkeyVAEEnv environment.
    This wrapper flattens observations and adapts the API to Gymnasium standards.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, donkey_env: DonkeyVAEEnv):
        self.donkey_env = donkey_env
        self._step_count = 0
        obs_dim = int(np.prod(donkey_env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=donkey_env.action_space.low.astype(np.float32),
            high=donkey_env.action_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self.spec = EnvSpec(id="DonkeyVae-v0", max_episode_steps=MAX_EPISODE_STEPS)
        self.render_mode = "human"
        self.max_episode_steps = MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        self._step_count = 0
        if seed is not None:
            self.donkey_env.seed(seed)
        obs = self.donkey_env.reset()
        return self._flatten_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self.donkey_env.step(action)
        self._step_count += 1
        terminated = bool(done)
        truncated = self._step_count >= self.max_episode_steps
        obs = self._flatten_obs(obs)
        info = info or {}
        if truncated:
            info.setdefault("TimeLimit.truncated", True)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.donkey_env.render(mode=mode)

    def close(self):
        try:
            self.donkey_env.close()
        except Exception:
            pass

    @staticmethod
    def _flatten_obs(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)


def make_donkey_vae_env(device: torch.device, vae_path: str = None):
    if vae_path is None:
        vae_path = os.environ.get(
            "VAE_ARCHIVE_DIR", "DonkeyCarEnv/vae-level-0-dim-32.pkl"
        )
    vae = load_ae(vae_path, z_size=32)
    donkey_env = DonkeyVAEEnv(
        level=LEVEL,
        frame_skip=FRAME_SKIP,
        vae=vae,
        const_throttle=None,
        min_throttle=MIN_THROTTLE,
        max_throttle=MAX_THROTTLE,
        max_cte_error=MAX_CTE_ERROR,
        n_command_history=N_COMMAND_HISTORY,
        n_stack=N_STACK,
    )
    return GymnasiumDonkeyWrapper(donkey_env)
