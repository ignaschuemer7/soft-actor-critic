import torch
import numpy as np
from tqdm import tqdm

def random_agent_loop(env, num_episodes, writer, seed):
    """
    A simple loop for a random agent.
    """
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        if writer is not None:
            writer.add_scalar("RandomAgent/Reward", episode_reward, episode)

        if episode % 10 == 0:
            print(f"Episode {episode}: avg batch reward = {episode_reward:.3f}")
