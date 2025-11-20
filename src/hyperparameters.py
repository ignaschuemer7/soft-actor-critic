from dataclasses import dataclass, field
from typing import Tuple, Optional
from alpha_scheduler import *
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "identity": nn.Identity,
}


@dataclass
class QNetworkConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    init_std: float = 0.1
    hidden_layers_act: str = "relu"
    output_activation: str = "identity"
    output_activation_fn: nn.Module = field(init=False)
    hidden_layers_act_fn: nn.Module = field(init=False)

    def __post_init__(self):
        self.output_activation_fn = _ACTIVATIONS[self.output_activation]()
        self.hidden_layers_act_fn = _ACTIVATIONS[self.hidden_layers_act]()


@dataclass
class SACAlgorithmConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: AlphaScheduler = FixedAlpha(0.2)
    target_entropy: Optional[float] = None
    auto_entropy_tuning: bool = True

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    warming_steps: int = (
        1_000  # Number of steps to warm up the replay buffer with random actions
    )


@dataclass
class ReplayBufferConfig:
    capacity: int = 10_000


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    start_steps: int = 10_000
    update_after: int = 1_000
    update_every: int = 50
    gradient_steps_per_update: int = 1
    max_ep_len: int = 1_000
    seed: int = 0
    device: str = "cuda"


@dataclass
class LoggerConfig:
    agent_name: str = "SAC"
    log_dir: str = "logs/"


@dataclass
class SACConfig:
    sac: SACAlgorithmConfig = field(default_factory=SACAlgorithmConfig)
    net: QNetworkConfig = field(default_factory=QNetworkConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    def to_dict(self):
        """Get dictionary representation of the config"""
        sac_dict = self.sac.__dict__.copy()

        if isinstance(self.sac.alpha, AlphaScheduler):
            sac_dict["alpha"] = self.sac.alpha.to_dict()

        return {
            "sac": sac_dict,
            "net": self.net.__dict__,
            "buffer": self.buffer.__dict__,
            "train": self.train.__dict__,
        }


# Example usage:

# config = SACConfig()
# config.net.hidden_sizes = (256, 256, 256) # Only change specific parameters
# config.buffer.batch_size = 128

# agent = SACAgent(env, config=config)

if __name__ == "__main__":
    config = SACConfig()
    print(config.to_dict())
