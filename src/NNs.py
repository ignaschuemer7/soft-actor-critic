# SAC agent implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


# Network class
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, seed=None):
        super(QNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = build_mlp(state_size, hidden_sizes, action_size)

    def forward(self, state):
        out = self.net(state)
        return torch.squeeze(out, -1)  # Ensure output is of shape (batch_size,)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes,
        log_std_min=-20,
        log_std_max=2,
        seed=None,
        action_scale=1.0,
    ):
        super(PolicyNetwork, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = build_mlp(state_size, hidden_sizes, action_size * 2)
        self.action_scale = action_scale

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


if __name__ == "__main__":
    # Example usage
    state_size = 3
    action_size = 1
    hidden_sizes = [256, 256]
    q_net = QNetwork(state_size, action_size, hidden_sizes, seed=0)
    policy_net = PolicyNetwork(state_size, action_size, hidden_sizes, seed=0)
    sample_state = torch.randn(4, state_size)
    q_values = q_net(sample_state)
    actions, log_probs = policy_net.sample_action(sample_state)
    print("Q-values:", q_values)
    print("Actions:", actions)
