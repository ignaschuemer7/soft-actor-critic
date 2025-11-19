import random
from collections import deque, namedtuple
from typing import List, Optional
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Experience Replay Buffer for storing and sampling transitions.
        Parameters:
            capacity (int): Maximum number of transitions to store.
            seed (int, optional): Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store a transition in the replay buffer."""
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Not enough samples in the replay buffer to sample {batch_size} transitions. \
                Current size: {len(self.memory)}"
            )
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
