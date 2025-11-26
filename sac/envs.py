import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional


"""
"reward_description": "Agent always receives the same constant reward at every step regardless of the action taken. 
The episode terminates after a fixed number of steps.",
"goal_description": "Validates SAC's stability, value estimation, and entropy maximization behavior when Q-values are nearly identical for all actions. 
A correct implementation should learn a high-entropy policy without diverging or collapsing the critic."
"""


class ConstantRewardEnv(gym.Env):

    def __init__(self, reward: float = 1.0, max_steps: int = 1):
        super().__init__()
        self.constant_reward = float(reward)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.current_step = 0
        self.episode_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        observation = np.zeros(1, dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        observation = np.zeros(1, dtype=np.float32)
        reward = self.constant_reward
        self.episode_reward += reward
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step
            }
        return observation, reward, terminated, truncated, info


"""
"reward_description": "Agent receives highest reward when its continuous action is near a target value and lower reward as it moves away, following a simple quadratic shape. 
Episodes are one or a few steps long so the problem reduces to a continuous bandit.",
"goal_description": "Checks that SAC's actor and critic can learn an accurate continuous Q-function and a policy centered on the optimal action. 
It also tests temperature tuning and reward scaling on a well-conditioned, analytically simple task.",
"""


class QuadraticActionRewardEnv(gym.Env):
    """One-step continuous bandit with a quadratic reward around a target action."""

    def __init__(
        self,
        target: float = 0.5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        max_steps: int = 1,
    ):
        super().__init__()
        self.target = float(target)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.current_step = 0
        self.episode_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        observation = np.zeros(1, dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        a = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        reward = -((a - self.target) ** 2)
        self.episode_reward += reward
        observation = np.zeros(1, dtype=np.float32)
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"action": a}
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step
            }
        return observation, reward, terminated, truncated, info


"""
"reward_description": "At each step the observation is pure noise, while reward is +1 for actions within a small band around zero and -1 otherwise. 
The episode ends after a fixed number of steps with no terminal bonus.",
"goal_description": "Ensures SAC can marginalize over irrelevant observation noise and still learn the globally optimal action distribution. 
It also probes robustness of entropy and value estimates when state features carry no information about returns.",
"""


class RandomObsBinaryRewardEnv(gym.Env):
    """Random observations; reward depends only on action magnitude."""

    def __init__(self, obs_dim: int = 4, threshold: float = 0.2, max_steps: int = 1):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.threshold = float(threshold)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.current_step = 0
        self.episode_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        # observation = self.np_random.standard_normal(self.obs_dim).astype(np.float32)
        # uniform noise instead of normal
        observation = self.np_random.uniform(
            low=-1.0, high=1.0, size=self.obs_dim
        ).astype(np.float32)
        self.episode_reward = 0.0
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        a = float(action[0])
        reward = 1.0 if abs(a) <= self.threshold else -1.0
        # observation = self.np_random.standard_normal(self.obs_dim).astype(np.float32)
        # uniform noise instead of normal
        observation = self.np_random.uniform(
            low=-1.0, high=1.0, size=self.obs_dim
        ).astype(np.float32)
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"action": a}
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step
            }
        return observation, reward, terminated, truncated, info


"""
"reward_description": "The agent controls a 1D point mass with continuous actions that move it toward a goal position, receiving a small negative step penalty and a positive bonus when reaching the goal. 
Episodes end on reaching the goal or after a maximum number of steps.",
"goal_description": "Tests SAC's ability to handle multi-step credit assignment and continuous control with discounted returns. 
It also checks that target networks, bootstrapping, and entropy terms work together to learn a smooth, goal-directed policy.",
"""


class OneDPointMassReachEnv(gym.Env):
    """1D point mass moves with continuous actions to reach a goal position."""

    def __init__(
        self,
        start_pos: float = 0.0,
        goal_pos: float = 1.0,
        max_steps: int = 50,
        dt: float = 1.0,
        action_low: float = -0.1,
        action_high: float = 0.1,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        goal_tolerance: float = 0.05,
    ):
        super().__init__()
        self.start_pos = float(start_pos)
        self.goal_pos = float(goal_pos)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.goal_tolerance = float(goal_tolerance)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.current_step = 0
        self.pos = 0.0
        self.episode_reward = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.pos = self.start_pos
        observation = np.array([self.pos], dtype=np.float32)
        self.episode_reward = 0.0
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        a = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )
        self.pos += a * self.dt

        reward = self.step_penalty
        reached_goal = abs(self.pos - self.goal_pos) <= self.goal_tolerance
        if reached_goal:
            reward += self.goal_reward
        self.episode_reward += reward

        terminated = reached_goal
        truncated = self.current_step >= self.max_steps
        observation = np.array([self.pos], dtype=np.float32)
        info = {"action": a}
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step
            }
        return observation, reward, terminated, truncated, info
