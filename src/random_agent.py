import numpy as np
import gymnasium as gym
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


def random_agent_loop(
    env: gym.Env,
    num_episodes: int,
    writer: Optional[SummaryWriter] = None,
    show_progress: bool = False,
    batch_size: int = 10,
    seed: Optional[int] = None,
) -> None:
    """
    Runs episodes with uniformly random actions in continuous (Box) action spaces.
    """
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(
            "random_continuous_agent_main_loop requires a continuous (Box) action space."
        )

    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

    num_batches = int(np.ceil(num_episodes / batch_size))
    for batch_idx in range(num_batches):
        batch_rewards = []
        for episode_in_batch in range(batch_size):
            global_episode = batch_idx * batch_size + episode_in_batch
            if global_episode >= num_episodes:
                break

            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            batch_rewards.append(total_reward)
            if writer is not None:
                writer.add_scalar(
                    "RandomAgent/EpisodeReward", total_reward, global_episode
                )

        if show_progress and batch_rewards:
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            print(
                f"Episode {batch_idx * batch_size}: avg batch reward = {avg_reward:.3f}"
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5")
    writer = SummaryWriter(log_dir="runs/Pendulum")
    random_agent_loop(
        env,
        num_episodes=5000,
        show_progress=True,
        batch_size=5,
        seed=42,
        writer=writer,
    )
