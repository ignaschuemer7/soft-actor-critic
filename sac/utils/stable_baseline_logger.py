from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from sac.utils.logger_utils import save_lengths, save_rewards
import os


class EpisodeLoggerSB3(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        max_episodes: int,
        save_dir: str = "",
        save_npy: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.writer = writer
        self.max_episodes = max_episodes
        self.save_dir = save_dir
        self.save_npy = save_npy

        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for idx, done in enumerate(dones):
            self.current_episode_reward += rewards[idx]
            self.current_episode_length += 1

            if done:  # Episode ended (terminated OR truncated)
                ep_r = float(self.current_episode_reward)
                ep_l = int(self.current_episode_length)

                # Log to TensorBoard
                self.writer.add_scalar("Episode/Reward", ep_r, self.episode_count)
                self.writer.add_scalar("Episode/Length", ep_l, self.episode_count)

                if self.verbose:
                    print(
                        f"[Episode {self.episode_count}] Reward={ep_r}, Length={ep_l}"
                    )

                # Save in memory
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)

                # Reset episode counters
                self.current_episode_reward = 0.0
                self.current_episode_length = 0

                self.episode_count += 1

                # Early stop if reached max episodes
                if self.episode_count >= self.max_episodes:
                    if self.save_npy:
                        # Ensure save directory exists
                        os.makedirs(self.save_dir, exist_ok=True)
                        # Save rewards and lengths as .npy files
                        save_rewards(self.save_dir, self.episode_rewards)
                        save_lengths(self.save_dir, self.episode_lengths)
                    if self.verbose:
                        print(f"Saved episode rewards and lengths to {self.save_dir}")

                    print("Reached max episodes → early stopping training.")
                    return False  # <- SB3 stops training

        return True
