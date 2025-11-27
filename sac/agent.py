import torch
import numpy as np

from .replay_buffer import ReplayBuffer, Transition
from .models import QNetwork, PolicyNetwork
from .utils.experiment_logger import ExperimentLogger
import torch.optim as optim
import gymnasium as gym
from typing import Any, Dict, Optional
from copy import deepcopy
import pprint
from collections import deque
from tqdm import tqdm
import random
import os

from torch.utils.tensorboard import SummaryWriter

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
#         L_π = 1/|B| * (α * log_pi_reparam - min(Qφ1(s, a_reparam), Qφ2(s, a_reparam)))

#     (Optional) Update Entropy Temperature α:
#       If α is learned automatically:
#         Update α by minimizing:
#           L_α = -α * (log_pi_reparam + target_entropy)

#     Update target Q-networks:
#       φ1_target = ρ * φ1_target + (1 - ρ) * φ1
#       φ2_target = ρ * φ2_target + (1 - ρ) * φ2


class SAC:
    def __init__(self, env: gym.Env, config: dict):

        self.env = env
        self.config = config
        self.device = torch.device(self.config["train"]["device"])
        self._set_seed(self.config["train"]["seed"])

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.config["buffer"]["capacity"])

        # Initialize Networks
        self.obs_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self._init_policy_network()
        self._init_q_networks()

        # Initialize Optimizers
        self._init_optimizers()

        # Entropy
        self.target_entropy = -float(self.action_size)
        if self.config["sac"]["auto_entropy_tuning"]:
            # the log of alpha is optimized
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.config["sac"]["alpha_lr"]
            )
            # Convert log_alpha to alpha --> alpha = e^(log_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.config["sac"]["alpha"]).to(self.device)

        self.env_name = self.config["logger"]["env_name"] or self._infer_env_name(env)
        self.agent_name = self.config["logger"]["agent_name"] or self.__class__.__name__
        self.logger = (
            ExperimentLogger(
                self.config["logger"],
                env_name=self.env_name,
                agent_name=self.agent_name,
            )
            if self.config["logger"]["enabled"]
            else None
        )

    def _init_q_networks(self) -> None:
        """Initialize Q-Networks and their Target Networks."""
        self.q_net1 = QNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config["q_net"]["hidden_sizes"],
            hidden_activations=self.config["q_net"]["hidden_layers_act"],
            output_activation=self.config["q_net"]["output_activation"],
            seed=self.config["train"]["seed"],
        ).to(self.device)
        self.q_net2 = QNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config["q_net"]["hidden_sizes"],
            hidden_activations=self.config["q_net"]["hidden_layers_act"],
            output_activation=self.config["q_net"]["output_activation"],
            seed=self.config["train"]["seed"],
        ).to(self.device)
        self.q_net1_target = deepcopy(self.q_net1).to(self.device)
        self.q_net2_target = deepcopy(self.q_net2).to(self.device)

    def _init_policy_network(self) -> None:
        """Initialize Policy Network."""
        self.policy_net = PolicyNetwork(
            obs_size=self.obs_size,
            action_size=self.action_size,
            hidden_sizes=self.config["policy_net"]["hidden_sizes"],
            log_std_min=self.config["policy_net"]["log_std_min"],
            log_std_max=self.config["policy_net"]["log_std_max"],
            action_scale=self.config["policy_net"]["action_scale"],
            hidden_activations=self.config["policy_net"]["hidden_layers_act"],
            output_activation=self.config["policy_net"]["output_activation"],
            seed=self.config["train"]["seed"],
        ).to(self.device)

    def _init_optimizers(self) -> None:
        """Initialize Optimizers for Networks."""
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config["sac"]["actor_lr"]
        )
        self.q1_optimizer = optim.Adam(
            self.q_net1.parameters(), lr=self.config["sac"]["critic_lr"]
        )
        self.q2_optimizer = optim.Adam(
            self.q_net2.parameters(), lr=self.config["sac"]["critic_lr"]
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
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
        # Warn if warming_steps is greater than capacity --> it will never train
        if self.config["train"]["warming_steps"] > self.config["buffer"]["capacity"]:
            print("Warning: warming_steps is greater than replay buffer capacity.")
        return len(self.replay_buffer) >= self.config["train"]["warming_steps"]

    def sample_batch(self) -> Transition:
        """Draw a mini-batch of transitions from the replay buffer."""
        transitions = self.replay_buffer.sample(self.config["train"]["batch_size"])
        batch = Transition(*zip(*transitions))

        states = torch.as_tensor(np.stack(batch.state), dtype=torch.float32).to(
            self.device
        )
        actions = torch.as_tensor(np.stack(batch.action), dtype=torch.float32).to(
            self.device
        )
        rewards = torch.as_tensor(np.array(batch.reward), dtype=torch.float32).to(
            self.device
        )
        next_states = torch.as_tensor(
            np.stack(batch.next_state), dtype=torch.float32
        ).to(self.device)
        dones = torch.as_tensor(np.array(batch.done), dtype=torch.float32).to(
            self.device
        )

        return Transition(
            state=states,
            action=actions,
            reward=rewards,
            next_state=next_states,
            done=dones,
        )

    def compute_target_q_values(
        self,
        rewards: Any,
        dones: Any,
        next_states: Any,
    ) -> Any:
        """Compute target Q-values using target critics, next actions, and entropy term."""
        with torch.no_grad():
            alpha = self.alpha.detach()
            next_actions, next_log_pi = self.policy_net.sample_action(next_states)
            target_q1 = self.q_net1_target(next_states, next_actions)
            target_q2 = self.q_net2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2)
            target_q_values = rewards + self.config["sac"]["gamma"] * (1 - dones) * (
                min_target_q - alpha * next_log_pi
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
        alpha = self.alpha.detach()
        policy_loss = (alpha * log_pi - min_q_values).mean()

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
        if self.config["sac"]["auto_entropy_tuning"]:
            # Compute alpha loss using the log of alpha (to ensure alpha is positive)
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
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
                self.config["sac"]["tau"] * param.data
                + (1.0 - self.config["sac"]["tau"]) * target_param.data
            )

        # Update Q-network 2 target
        for target_param, param in zip(
            self.q_net2_target.parameters(), self.q_net2.parameters()
        ):
            target_param.data.copy_(
                self.config["sac"]["tau"] * param.data
                + (1.0 - self.config["sac"]["tau"]) * target_param.data
            )

    def training_step(self):
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

    def run_training_loop(
        self,
        num_episodes: int,
        logger: Optional[ExperimentLogger] = None,
        tqdm_disable: bool = False,
        print_rewards: bool = False,
    ) -> Dict[str, float]:
        """Main environment-interaction loop that collects data and triggers updates."""

        active_logger = logger or self.logger
        total_episodes = 0
        total_steps = 0
        returns_window = deque(maxlen=100)
        best_avg_return = -float("inf")
        for episode in tqdm(range(num_episodes), disable=tqdm_disable):
            state, _ = self.env.reset()
            done = False
            episode_return = 0.0
            total_episodes += 1
            episode_steps = 0
            while not done:
                # Select action according to current policy
                action = self.select_action(state)
                # Step the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Store transition in replay buffer
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                episode_steps += 1
                total_steps += 1
                # Perform training step if enough data is available
                if self.can_update():
                    for _ in range(self.config["train"]["gradient_steps_per_update"]):
                        self.training_step()
                if active_logger is not None and self.config["logger"]["log_q_values"]:
                    self._log_q_values(
                        states=torch.FloatTensor(state).unsqueeze(0).to(self.device),
                        actions=torch.FloatTensor(action).unsqueeze(0).to(self.device),
                        logger=active_logger,
                        step=total_steps,
                    )

            returns_window.append(episode_return)
            avg_return = np.mean(returns_window)
            best_avg_return = max(best_avg_return, avg_return)

            if active_logger is not None and self.config["logger"]["log_episode_stats"]:
                active_logger.log_episode_metrics(
                    episode_idx=episode,
                    reward=episode_return,
                    length=episode_steps,
                )
            if print_rewards:
                print(
                    f"Episode {episode}, Return: {episode_return:.2f}, Average Return(last 100 episodes): {avg_return:.2f}"
                )
        metrics = {
            "total_episodes": total_episodes,
            "best_avg_return": best_avg_return,
            "final_avg_return": avg_return,
        }
        if active_logger is not None:
            active_logger.log_hparams(self.config, metrics)

        if self.config["logger"]["save_model"]["enabled"]:
            save_path = self.config["logger"]["save_model"]["path"]
            if save_path is None:
                save_path = active_logger.run_dir
            else:
                os.makedirs(save_path, exist_ok=True)

            model_path = os.path.join(save_path, "sac_agent.pth")
            self.save_agent(model_path)
            print(f"Agent saved to {model_path}")
        
        # save rewards and lengths as txt files
        if active_logger is not None:
            active_logger.save()

        return metrics

    def eval_agent(
        self,
        num_episodes: int,
        render_mode: Optional[str] = None,
        tqdm_disable: bool = False,
        print_returns: bool = False,
        writer: Optional[SummaryWriter] = None,
    ) -> float:
        """Evaluate the agent's performance over a number of episodes."""
        # If a render_mode is specified for evaluation, and it's different from the agent's env,
        # try to create a new environment.
        eval_env = self._get_render_environment(render_mode)

        total_return = 0.0
        for episode in tqdm(range(num_episodes), disable=tqdm_disable):
            state, _ = eval_env.reset()
            done = False
            episode_return = 0.0
            length = 0
            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                state = next_state
                episode_return += reward
                length += 1
            total_return += episode_return
            if print_returns:
                print(f"Evaluation Episode {episode}, Return: {episode_return:.2f}")
            if writer is not None:
                writer.add_scalar("Eval/Episode/Return", episode_return, episode)
                writer.add_scalar("Eval/Episode/Length", length, episode)
        avg_return = total_return / num_episodes
        if print_returns:
            print(f"Average Return over {num_episodes} episodes: {avg_return:.2f}")

        # Close the evaluation environment if it was newly created
        if eval_env is not self.env:
            eval_env.close()

        return avg_return

    def _get_render_environment(self, render_mode: Optional[str]) -> gym.Env:
        """Get an environment suitable for rendering during evaluation."""
        if (
            render_mode is not None
            and getattr(self.env, "render_mode", None) != render_mode
        ):
            if self.env.spec and self.env.spec.id:
                try:
                    print(
                        f"Creating new environment for evaluation with render_mode='{render_mode}'"
                    )
                    eval_env = gym.make(self.env.spec.id, render_mode=render_mode)
                    # Seed the new env for reproducibility if a seed is available in config.
                    seed = self.config["train"].get("seed")
                    if seed is not None:
                        eval_env.reset(seed=seed)
                        eval_env.action_space.seed(seed)
                except Exception as e:
                    print(
                        f"Warning: Failed to create new env for rendering: {e}. Using original env."
                    )
                    eval_env = self.env
            else:
                print(
                    "Warning: Cannot create new env for rendering as env.spec.id is not available. Using original env."
                )
                eval_env = self.env
        return eval_env

    def _log_q_values(
        self, states: Any, actions: Any, logger: ExperimentLogger, step: int
    ) -> None:
        """Log Q-values from both Q-networks for given states and actions."""
        with torch.no_grad():
            q1_values = self.q_net1(states, actions)
            q2_values = self.q_net2(states, actions)
            logger.log_q_values(q1_values.mean().item(), q2_values.mean().item(), step)

    def _infer_env_name(self, env: gym.Env) -> str:
        if getattr(env, "spec", None) is not None and getattr(env.spec, "id", None):
            return env.spec.id
        return env.__class__.__name__

    def show_config(self, indent: int = 4) -> None:
        """Print the current configuration in a readable format."""
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.config)

    def print_net_arqhitectures(self) -> None:
        """Print the architectures of the networks."""
        print("Policy Network Architecture:")
        print(self.policy_net)
        print("\nQ-Network 1 Architecture:")
        print(self.q_net1)
        print("\nQ-Network 2 Architecture:")
        print(self.q_net2)

    def save_agent(self, filepath: str) -> None:
        """Save the agent's networks and optimizers to a file."""
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "q_net1_state_dict": self.q_net1.state_dict(),
            "q_net2_state_dict": self.q_net2.state_dict(),
            "q_net1_target_state_dict": self.q_net1_target.state_dict(),
            "q_net2_target_state_dict": self.q_net2_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
            "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
        }
        if self.config["sac"]["auto_entropy_tuning"]:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, filepath)
        # call self.logger.save to save any additional logger info (buffers, etc.)

    def load_agent(self, filepath: str) -> None:
        """Load the agent's networks and optimizers from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.q_net1.load_state_dict(checkpoint["q_net1_state_dict"])
        self.q_net2.load_state_dict(checkpoint["q_net2_state_dict"])
        self.q_net1_target.load_state_dict(checkpoint["q_net1_target_state_dict"])
        self.q_net2_target.load_state_dict(checkpoint["q_net2_target_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer_state_dict"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer_state_dict"])
        if self.config["sac"]["auto_entropy_tuning"]:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
            self.alpha = self.log_alpha.exp()
        # call self.logger.load to load any additional logger info (buffers, etc.)
