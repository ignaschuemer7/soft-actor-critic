import argparse
import yaml
import gymnasium as gym
import torch

from sac.agent import SAC
from sac.envs import *


def main(args):
    """
    Main function to run the training.
    """
    # Load hyperparameters from config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Configuration loaded:")
    print(config)

    # Initialize environment
    env_name = config["logger"]["env_name"]

    if env_name == "ConstantRewardEnv":
        env = ConstantRewardEnv()
    elif env_name == "QuadraticActionRewardEnv":
        env = QuadraticActionRewardEnv()
    elif env_name == "RandomObsBinaryRewardEnv":
        env = RandomObsBinaryRewardEnv()
    elif env_name == "OneDPointMassReachEnv":
        env = OneDPointMassReachEnv()
    else:
        env = gym.make(env_name)

    # Initialize agent
    agent = SAC(env, config)

    print("Agent initialized. Starting training...")
    agent.run_training_loop(
        num_episodes=2000
    )  # Run for a small number of episodes for testing

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reacher_env.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
