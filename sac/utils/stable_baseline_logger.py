from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class RewardLoggingCallback(BaseCallback):
    def __init__(self, writer: SummaryWriter, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 saves episode info in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                self.writer.add_scalar("Episode/Reward", ep_reward, self.num_timesteps)
                self.writer.add_scalar("Episode/Length", ep_length, self.num_timesteps)

                if self.verbose:
                    print(f"Step={self.num_timesteps}, Reward={ep_reward}")

        return True
