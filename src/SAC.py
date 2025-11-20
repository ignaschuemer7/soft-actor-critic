# Initialize networks:
#   Policy network πθ(a|s) with parameters θ
#   Two Q-networks Qφ1(s,a) and Qφ2(s,a) with parameters φ1, φ2
#   Two target Q-networks Qφ1_target(s,a) and Qφ2_target(s,a) with parameters φ1_target, φ2_target
#   (Optional) Entropy temperature α (can be learned automatically)

# Initialize replay buffer D
# Warming up the replay buffer D with random actions for a certain number of steps

# Loop for each training iteration:
#   Sample a state s from the environment
#   Sample an action a from the policy πθ(a|s)
#   Execute action a in the environment, observe reward r, next state s', and done flag d
#   Store transition (s, a, r, s', d) in replay buffer D

#   If enough transitions in D:
#     Sample a mini-batch of transitions (s, a, r, s', d) from D with size |B|

#     Update Q-networks:
#       Compute target Q-values:
#         Sample a_next ~ πθ(a|s')
#         log_pi_next = log(πθ(a_next|s'))
#         q_target = r + γ * (1 - d) * (min(Qφ1_target(s', a_next), Qφ2_target(s', a_next)) - α * log_pi_next)

#       Update Q-network parameters φ1, φ2 by minimizing:
#         L_Q = 1/|B| * (MSE(Qφ1(s,a), q_target) + MSE(Qφ2(s,a), q_target))

#     Update Policy network:
#       Sample a_reparam ~ πθ(a|s) (using reparameterization trick)
#       log_pi_reparam = log(πθ(a_reparam|s))

#       Update policy parameters θ by minimizing:
#         L_π = 1/|B| * (min(Qφ1(s, a_reparam), Qφ2(s, a_reparam)) - α * log_pi_reparam)

#     (Optional) Update Entropy Temperature α:
#       If α is learned automatically:
#         Update α by minimizing:
#           L_α = -α * (log_pi_reparam + target_entropy)

#     Update target Q-networks:
#       φ1_target = ρ * φ1_target + (1 - ρ) * φ1
#       φ2_target = ρ * φ2_target + (1 - ρ) * φ2


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np
from hyperparameters import SACConfig
from replay_buffer import ReplayBuffer
from NNs import QNetwork, PolicyNetwork
import torch.optim as optim
import gymnasium as gym
from copy import deepcopy
import pprint


class SAC:
    def __init__(self, env: gym.Env, config: SACConfig):

        self.env = env
        self.config = config
        self.device = torch.device(self.config.train.device)
        self._set_seed(self.config.train.seed)
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer.capacity)
        # Initialize Networks
        self.obs_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self._init_policy_network()
        self._init_q_networks()
        # Initialize Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config.sac.actor_lr
        )
        self.q1_optimizer = optim.Adam(
            self.q_net1.parameters(), lr=self.config.sac.critic_lr
        )
        self.q2_optimizer = optim.Adam(
            self.q_net2.parameters(), lr=self.config.sac.critic_lr
        )
        self.alpha = self.config.sac.alpha

    def _init_q_networks(self):
        # Initialize Q-Networks and Target Networks
        self.q_net1 = QNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config.q_net.hidden_sizes,
        ).to(self.device)
        self.q_net2 = QNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config.q_net.hidden_sizes,
        ).to(self.device)
        self.q_net1_target = deepcopy(self.q_net1).to(self.device)
        self.q_net2_target = deepcopy(self.q_net2).to(self.device)

    def _init_policy_network(self):
        # Initialize Policy Network
        self.policy_net = PolicyNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config.policy_net.hidden_sizes,
            log_std_min=self.config.policy_net.log_std_min,
            log_std_max=self.config.policy_net.log_std_max,
            action_scale=self.config.policy_net.action_scale,
        ).to(self.device)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

    def show_config(self, indent: int = 4):
        """Print the current configuration in a readable format."""
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.config.to_dict())

    def print_net_arqhitectures(self):
        """Print the architectures of the networks."""
        print("Policy Network Architecture:")
        print(self.policy_net)
        print("\nQ-Network 1 Architecture:")
        print(self.q_net1)
        print("\nQ-Network 2 Architecture:")
        print(self.q_net2)


if __name__ == "__main__":
    # Example usage
    config = SACConfig()
    env = gym.make("Pendulum-v1")
    sac_agent = SAC(env, config)
    print("SAC agent initialized.")
    sac_agent.show_config()
    sac_agent.print_net_arqhitectures()
