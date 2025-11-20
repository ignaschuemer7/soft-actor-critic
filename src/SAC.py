from typing import Any, Dict, Optional

from hyperparameters import SACConfig

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



class SAC:
    def __init__(self, state_dim: int, action_dim: int, config: Optional[SACConfig] = None):
        """TODO: initialize networks, optimizers, replay buffer, and alpha scheduler."""
        pass

    def initialize_networks(self) -> None:
        """TODO: create policy network, twin Q networks, and corresponding target networks."""
        pass

    def initialize_replay_buffer(self) -> None:
        """TODO: set up replay buffer storage."""
        pass

    def warmup_replay_buffer(self, env: Any, steps: int) -> None:
        """TODO: prefill the replay buffer using random actions for a fixed number of steps."""
        pass

    def select_action(self, state: Any, deterministic: bool = False) -> Any:
        """TODO: sample or choose an action from the policy given the current state."""
        pass

    def store_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """TODO: push a transition tuple (s, a, r, s', d) into the replay buffer."""
        pass

    # This could be simple conditional inside another func instead of a function
    def can_update(self) -> bool:
        """TODO: return True when the replay buffer contains enough samples to learn."""
        pass

    def sample_batch(self) -> Dict[str, Any]:
        """TODO: draw a mini-batch of transitions from the replay buffer."""
        pass

    def compute_target_q_values(
        self,
        rewards: Any,
        dones: Any,
        next_states: Any,
    ) -> Any:
        """TODO: compute target Q-values using target critics, next actions, and entropy term."""
        pass

    def update_q_networks(
        self,
        states: Any,
        actions: Any,
        target_q_values: Any,
    ) -> Dict[str, float]:
        """TODO: update Q-network parameters by minimizing critic loss against targets."""
        pass

    def update_policy_network(self, states: Any) -> Dict[str, float]:
        """TODO: update policy parameters via reparameterized action samples."""
        pass

    # This one in the pseudo-code is optional
    def update_entropy_temperature(
        self,
        log_pi: Any,
        step: int = 0,
    ) -> Dict[str, float]:
        """TODO: optionally tune alpha toward target entropy."""
        pass

    def soft_update_target_networks(self) -> None:
        """TODO: polyak-average target networks toward online critic parameters."""
        pass

    def training_step(self, env_step: int) -> Dict[str, float]:
        """TODO: run one gradient update of critics, policy, temperature, and targets."""
        pass

    def run_training_loop(self, env: Any, total_steps: int) -> None:
        """TODO: main environment-interaction loop that collects data and triggers updates."""
        pass
