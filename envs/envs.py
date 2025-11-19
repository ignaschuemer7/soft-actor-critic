import gymnasium as gym
import numpy as np
from gymnasium import spaces

"""
Acción única, sin observación, un solo paso, recompensa constante
- Acciones disponibles: 1 (única acción)
- Observaciones: constante 0
- Duración: 1 paso de tiempo (episodio de una sola transición)
- Recompensa: +1 en cada episodio
"""


class ConstantRewardEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)  # una única acción posible: 0
        self.observation_space = spaces.Discrete(1)  # observación constante: 0

    def _get_obs(self):
        return 0

    def _get_info(self):
        return {}

    def step(self, action):
        reward = 1  # recompensa constante
        observation = self._get_obs()
        terminated = True  # el episodio termina después de un solo paso
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


"""
Acción única, observación aleatoria, un solo paso, recompensa dependiente de
la observación
- Acciones disponibles: 1 (única acción)
- Observaciones: aleatorias, con valor +1 o -1
- Duración: 1 paso de tiempo
- Recompensa: coincide con la observación (+1 o -1)
"""


class RandomObsBinaryRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(2)
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(0, 2)  # 0 o 1
        return self.state, {}

    def step(self, action):
        # recompensa basada en el estado actual
        reward = 1 if self.state == 1 else -1
        terminated = True
        truncated = False
        info = {}

        next_obs = self.state
        return next_obs, reward, terminated, truncated, info


"""
Acción única, observación determinista, dos pasos, recompensa diferida
- Acciones disponibles: 1 (única acción)
- Observaciones: en el primer paso se observa 0; en el segundo paso se observa 1
- Duración: 2 pasos por episodio
- Recompensa: 0 en el primer paso, +1 al final del episodio
"""


class TwoStepDelayedRewardEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(1)  # una única acción posible: 0
        self.observation_space = spaces.Discrete(2)  # observaciones: 0 o 1
        self.current_step = 0

    def _get_obs(self):
        return self.current_step  # observación depende del paso actual

    def _get_info(self):
        return {}

    def step(self, action):
        if self.current_step == 0:
            reward = 0  # recompensa en el primer paso
            self.current_step += 1
            terminated = False
        else:
            reward = 1  # recompensa al final del episodio
            terminated = True
        observation = self._get_obs()
        truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
