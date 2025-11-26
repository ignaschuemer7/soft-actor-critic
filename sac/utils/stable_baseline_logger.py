from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


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

