import sys
from pathlib import Path

# Add project roots to import path
REPO_ROOT = Path("..").resolve()
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "learning-to-drive-in-5-minutes"))

import sys
import numpy as np
import gymnasium as gym
import torch

# Alias gymnasium as gym for donkey_gym, which expects the old API
sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)
sys.modules.setdefault("gym.envs", gym.envs)
sys.modules.setdefault("gym.envs.registration", gym.envs.registration)
sys.modules.setdefault("gym.utils", gym.utils)

import os
import io
import zipfile
from typing import Any
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
import cv2
from gymnasium.envs.registration import EnvSpec
from donkey_gym.envs.vae_env import DonkeyVAEEnv
from config import LEVEL, FRAME_SKIP, MIN_THROTTLE, MAX_THROTTLE, MAX_CTE_ERROR, ROI
from vae.data_loader import preprocess_input

# Configure simulator path/port here
# os.environ.setdefault("DONKEY_SIM_PATH", str(Path("/home/san/Documents/Ingenieria UdeSA/RL/DonkeySimLinux/donkey_sim.x86_64")))
os.environ.setdefault("DONKEY_SIM_PATH", str(Path("/home/san/Documents/Ingenieria UdeSA/RL/donkey_simulator/build_sdsandbox.x86_64")))
# Default Unity build listens on 9090; change if you launch with --port X
os.environ.setdefault("DONKEY_SIM_PORT", "9091")
# Optional: fail fast instead of waiting forever
os.environ.setdefault("DONKEY_WAIT_TIMEOUT", "30")
os.environ.setdefault("DONKEY_SKIP_WAIT", "1")

MAX_EPISODE_STEPS = 1000  # fallback horizon for wrappers expecting a finite episode length
vae_archive_dir = REPO_ROOT / "learning-to-drive-in-5-minutes" / "ae-32_mountain" / "archive"
print(f"Loading VAE weights from extracted archive dir: {vae_archive_dir}")
assert vae_archive_dir.exists(), "VAE archive folder not found."


def _load_torch_vae_checkpoint(extracted_archive: Path):
    """Load a torch checkpoint from an extracted torch archive folder."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for file_path in extracted_archive.rglob("*"):
            if file_path.is_file():
                # torch expects entries like archive/data.pkl, archive/data/0, etc.
                rel_path = file_path.relative_to(extracted_archive.parent)
                info = zipfile.ZipInfo(str(rel_path), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, file_path.read_bytes())
    buffer.seek(0)
    try:
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(buffer, map_location="cpu")
    return checkpoint


class TorchConvEncoder(torch.nn.Module):
    def __init__(self, z_size: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(),
        )
        self.proj = torch.nn.Linear(128 * 3 * 8, z_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.reshape(h.size(0), -1)
        return self.proj(h)


class TorchVAEWrapper:
    """Minimal VAE wrapper that matches the Donkey env API (encode + z_size)."""

    def __init__(self, archive_dir: Path):
        checkpoint = _load_torch_vae_checkpoint(archive_dir)
        z_size = int(checkpoint["data"]["z_size"])
        self.model = TorchConvEncoder(z_size)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()
        self.z_size = z_size
        self.normalization_mode = checkpoint["data"].get("normalization_mode", "rl")

    def encode(self, observation: Any):
        # If we somehow already receive a latent vector, pass it through
        obs_arr = np.asarray(observation)
        if obs_arr.ndim < 3 or obs_arr.shape[-1] != 3:
            return obs_arr.reshape(1, -1).astype(np.float32)

        h, w = obs_arr.shape[:2]
        # Crop if larger than ROI, otherwise resize up to the expected VAE input
        if h > ROI[3] and w > ROI[2]:
            cropped = obs_arr[int(ROI[1]): int(ROI[1] + ROI[3]), int(ROI[0]): int(ROI[0] + ROI[2])]
        else:
            cropped = cv2.resize(obs_arr, (ROI[2], ROI[3]), interpolation=cv2.INTER_AREA)

        processed = preprocess_input(cropped.astype(np.float32), mode=self.normalization_mode)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            z = self.model(tensor)
        return z.cpu().numpy()


class GymnasiumDonkeyWrapper(gym.Env):
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

    @staticmethod
    def _flatten_obs(obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)

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
            self.donkey_env.close_connection()
        finally:
            try:
                self.donkey_env.exit_scene()
            except Exception:
                pass


def make_donkey_vae_env():
    vae = TorchVAEWrapper(vae_archive_dir)
    donkey_env = DonkeyVAEEnv(
        level=LEVEL,
        frame_skip=FRAME_SKIP,
        vae=vae,
        const_throttle=None,
        min_throttle=MIN_THROTTLE,
        max_throttle=MAX_THROTTLE,
        max_cte_error=MAX_CTE_ERROR,
        n_command_history=0,
    )
    return GymnasiumDonkeyWrapper(donkey_env)

if __name__ == "__main__":
    env = make_donkey_vae_env()
    print(f"Env ready with obs shape {env.observation_space.shape} and action shape {env.action_space.shape}")