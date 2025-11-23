# SAC agent implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_size,
        action_size,
        hidden_sizes,
        hidden_activations=nn.ReLU,
        output_activation=nn.Identity,
        seed=None,
    ):
        super(QNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = build_mlp(
            obs_size + action_size,
            hidden_sizes,
            1,
            hidden_activations,
            output_activation,
        )
        self._init_weights_xavier()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        out = self.net(sa)
        return torch.squeeze(out, -1)  # Ensure output is of shape (batch_size,)

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def _init_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        obs_size,
        action_size,
        hidden_sizes,
        log_std_min=-20,
        log_std_max=2,
        seed=None,
        action_scale=1.0,
        hidden_activations=nn.ReLU,
        output_activation=nn.Identity,
    ):
        super(PolicyNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = build_mlp(
            obs_size,
            hidden_sizes,
            action_size * 2,
            hidden_activations,
            output_activation,
        )
        self.action_scale = action_scale
        self._init_weights_xavier()

    def forward(self, state):
        mu_log_std = self.net(state)
        mu, log_std = torch.chunk(mu_log_std, 2, dim=-1)  # Split into mean and log std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        pi_normal = torch.distributions.Normal(mu, std)
        z = pi_normal.rsample()
        action = torch.tanh(z) * self.action_scale
        log_prob = pi_normal.log_prob(z).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(axis=-1)
        return action, log_prob

    def deterministic_action(self, state):
        mu, _ = self.forward(state)
        action = torch.tanh(mu) * self.action_scale
        return action

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def _init_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def build_mlp(
    obs_size: int,
    hidden_sizes: List[int],
    action_size: int,
    hidden_activations: nn.Module = nn.Tanh,
    output_activation: nn.Module = nn.Identity,
) -> nn.Sequential:
    """
    Build a simple MLP network.

    Args:
        obs_size: Dimension of the observation space
        hidden_sizes: List of hidden layer sizes
        action_size: Number of actions
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer

    Returns:
        nn.Sequential: MLP model
    """
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty")

    layer_sizes = [obs_size] + hidden_sizes + [action_size]

    layers = []
    for i in range(len(layer_sizes) - 1):
        act = hidden_activations if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


if __name__ == "__main__":
    # Example usage
    obs_size = 3
    action_size = 1
    hidden_sizes = [256, 256]
    q_net = QNetwork(obs_size, action_size, hidden_sizes, seed=0)
    policy_net = PolicyNetwork(obs_size, action_size, hidden_sizes, seed=0)
    sample_state = torch.randn(4, obs_size)
    sample_action = torch.randn(4, action_size)
    q_values = q_net(sample_state, sample_action)
    actions, log_probs = policy_net.sample_action(sample_state)
    print("Q-values:", q_values)
    print("Actions:", actions)
