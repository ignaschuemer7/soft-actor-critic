import torch
import numpy as np
from hyperparameters import SACConfig
from replay_buffer import ReplayBuffer, Transition
from NNs import QNetwork, PolicyNetwork
import torch.optim as optim
import gymnasium as gym
from typing import Any, Dict
from copy import deepcopy
import pprint

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
        self._init_optimizers()

        # Entropy
        self.target_entropy = -float(self.action_size)
        if self.config.sac.auto_entropy_tuning:
            # the log of alpha is optimized
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.sac.alpha_lr)
            # Convert log_alpha to alpha --> alpha = e^(log_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.config.sac.alpha).to(self.device)


    def _init_q_networks(self) -> None:
        """Initialize Q-Networks and their Target Networks."""
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


    def _init_policy_network(self) -> None:
        """Initialize Policy Network."""
        self.policy_net = PolicyNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config.policy_net.hidden_sizes,
            log_std_min=self.config.policy_net.log_std_min,
            log_std_max=self.config.policy_net.log_std_max,
            action_scale=self.config.policy_net.action_scale,
        ).to(self.device)


    def _init_optimizers(self) -> None:
        """Initialize Optimizers for Networks."""
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config.sac.actor_lr
        )
        self.q1_optimizer = optim.Adam(
            self.q_net1.parameters(), lr=self.config.sac.critic_lr
        )
        self.q2_optimizer = optim.Adam(
            self.q_net2.parameters(), lr=self.config.sac.critic_lr
        )


    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)


    def store_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Push a transition tuple (s, a, r, s', d) into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    

    def warmup_replay_buffer(self, env: Any, steps: int) -> None:
        """Prefill the replay buffer using random actions for a fixed number of steps."""
        state, _ = env.reset()
        for _ in range(steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.store_transition(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = env.reset()


    def select_action(self, state: Any, deterministic: bool = False) -> Any:
        """Sample or choose an action from the policy given the current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            action = self.policy_net.deterministic_action(state_tensor)
        else:
            action, _ = self.policy_net.sample_action(state_tensor)
        return action.detach().cpu().numpy()[0]


    # This could be simple conditional inside another func instead of a function
    def can_update(self) -> bool:
        """Return True when the replay buffer contains enough samples to learn."""
        # Warn if update_after is greater than capacity --> it will never train
        if (self.config.train.update_after > self.config.buffer.capacity):
            print("Warning: update_after is greater than replay buffer capacity.")
        return len(self.replay_buffer) >= self.config.train.update_after


    def sample_batch(self) -> Transition:
        """Draw a mini-batch of transitions from the replay buffer."""
        transitions = self.replay_buffer.sample(self.config.train.batch_size)
        batch = Transition(*zip(*transitions))
        
        return Transition(
            state=torch.FloatTensor(batch.state).to(self.device),
            action=torch.FloatTensor(batch.action).to(self.device),
            reward=torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device),
            next_state=torch.FloatTensor(batch.next_state).to(self.device),
            done=torch.FloatTensor(batch.done).unsqueeze(1).to(self.device),
        )
        

    def compute_target_q_values(
        self,
        rewards: Any,
        dones: Any,
        next_states: Any,
    ) -> Any:
        """Compute target Q-values using target critics, next actions, and entropy term."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_pi = self.policy_net.sample_action(next_states)
            
            # Compute target Q-values from both target networks
            target_q1 = self.q_net1_target(next_states, next_actions)
            target_q2 = self.q_net2_target(next_states, next_actions)
            
            # Take minimum to reduce overestimation bias
            min_target_q = torch.min(target_q1, target_q2)
            
            # Compute target: r + γ * (1 - d) * (min_Q_target - α * log_π)
            target_q_values = rewards + self.config.sac.gamma * (1 - dones) * (
                min_target_q - self.alpha * next_log_pi
            )

        return target_q_values


    def update_q_networks(
        self,
        states: Any,
        actions: Any,
        target_q_values: Any,
    ):
        """Update Q-network parameters by minimizing critic loss against targets."""
        # Compute current Q-values
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)

        # Compute critic losses
        q1_loss = torch.nn.functional.mse_loss(current_q1, target_q_values)
        q2_loss = torch.nn.functional.mse_loss(current_q2, target_q_values)

        # Optimize Q-network 1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Optimize Q-network 2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()



    def update_policy_network(self, states: Any):
        """Update policy parameters via reparameterized action samples."""
        # Sample actions from the current policy
        actions, log_pi = self.policy_net.sample_action(states)

        # Compute Q-values for the sampled actions
        q1_values = self.q_net1(states, actions)
        q2_values = self.q_net2(states, actions)

        # Take minimum Q-value to reduce overestimation bias
        min_q_values = torch.min(q1_values, q2_values)

        # Compute policy loss
        policy_loss = (min_q_values - self.alpha * log_pi).mean()

        # Optimize policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # return log_pi for entropy temperature update
        return log_pi


    # This one in the pseudo-code is optional
    def update_entropy_temperature(
        self,
        log_pi: Any,
    ) -> Dict[str, float]:
        if self.config.sac.auto_entropy_tuning:
            # Compute alpha loss using the log of alpha (to ensure alpha is positive)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            # Optimize alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Update alpha value
            self.alpha = self.log_alpha.exp()
            return {"alpha_loss": alpha_loss.item(), "alpha": self.alpha.item()}
        else:
            return {}


    def soft_update_target_networks(self) -> None:
        """Polyak-average target networks toward online critic parameters."""
        # Update Q-network 1 target
        for target_param, param in zip(
            self.q_net1_target.parameters(), self.q_net1.parameters()
        ):
            target_param.data.copy_(
                self.config.sac.tau * param.data + (1.0 - self.config.sac.tau) * target_param.data
            )

        # Update Q-network 2 target
        for target_param, param in zip(
            self.q_net2_target.parameters(), self.q_net2.parameters()
        ):
            target_param.data.copy_(
                self.config.sac.tau * param.data + (1.0 - self.config.sac.tau) * target_param.data
            )


    def training_step(self, env_step: int):
        """Run one gradient update of critics, policy, temperature, and targets."""
        batch = self.sample_batch()

        # Compute target Q-values
        target_q_values = self.compute_target_q_values(
            rewards=batch.reward,
            dones=batch.done,
            next_states=batch.next_state,
        )

        # Update Q-networks
        self.update_q_networks(
            states=batch.state,
            actions=batch.action,
            target_q_values=target_q_values,
        )

        # Update Policy network
        log_pi = self.update_policy_network(states=batch.state)

        # Update Entropy Temperature
        self.update_entropy_temperature(log_pi=log_pi)

        # Soft update target networks
        self.soft_update_target_networks()


    def run_training_loop(self, env: Any, total_steps: int):
        """Main environment-interaction loop that collects data and triggers updates."""
        state, _ = env.reset()
        for env_step in range(total_steps):
            # Select action according to current policy
            action = self.select_action(state)

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            self.store_transition(state, action, reward, next_state, done)

            state = next_state

            if done:
                state, _ = env.reset()

            # Perform training step if enough data is available
            if self.can_update():
                for _ in range(self.config.train.gradient_steps_per_update):
                    self.training_step(env_step)


    def show_config(self, indent: int = 4) -> None:
        """Print the current configuration in a readable format."""
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.config.to_dict())


    def print_net_arqhitectures(self) -> None:
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
    env = gym.make("InvertedPendulum-v5")
    sac_agent = SAC(env, config)
    print("SAC agent initialized.")
    sac_agent.show_config()
    sac_agent.print_net_arqhitectures()
