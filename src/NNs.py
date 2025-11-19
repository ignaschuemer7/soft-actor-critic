# SAC agent implementation 

import torch
import torch.nn as nn

from utils import *

# Network class
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super(QNetwork, self).__init__()
        self.net = build_mlp(state_size, hidden_sizes, action_size)

    def forward(self, state):
        return self.net(state)
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = build_mlp(state_size, hidden_sizes, action_size * 2)

    def forward(self, state):
        mu_log_std = self.net(state)
        mu, log_std = torch.chunk(mu_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std