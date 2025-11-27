from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import os

class RewardLoggingCallback(BaseCallback):
    def __init__(self, writer: SummaryWriter, num_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.writer = writer
        self.episode_count = 0 
        self.num_episodes = num_episodes

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and self.episode_count < self.num_episodes:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                self.writer.add_scalar("Episode/Reward", ep_reward, self.episode_count)
                self.writer.add_scalar("Episode/Length", ep_length, self.episode_count)

                if self.verbose:
                    print(f"Episode={self.episode_count}, Reward={ep_reward}, Length={ep_length}")

                self.episode_count += 1

        return True

class RobustEpisodeLogger(BaseCallback):
    def __init__(
        self,
        writer: SummaryWriter,
        max_episodes: int,
        save_dir: str = "",
        save_txt: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.writer = writer
        self.max_episodes = max_episodes
        self.save_dir = save_dir
        self.save_txt = save_txt

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
                    print(f"[Episode {self.episode_count}] Reward={ep_r}, Length={ep_l}")

                # Save in memory
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)

                # Reset episode counters
                self.current_episode_reward = 0.0
                self.current_episode_length = 0

                self.episode_count += 1

                # Early stop if reached max episodes
                if self.episode_count >= self.max_episodes:
                    if self.save_txt:
                        os.makedirs(self.save_dir, exist_ok=True)
                        with open(f"{self.save_dir}/episode_rewards.txt", "w") as f:
                            for r in self.episode_rewards:
                                f.write(f"{r}\n")
                        with open(f"{self.save_dir}/episode_lengths.txt", "w") as f:
                            for l in self.episode_lengths:
                                f.write(f"{l}\n")

                    print("Reached max episodes → early stopping training.")
                    return False  # <- SB3 stops training

        return True
