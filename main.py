import argparse
import yaml
import gymnasium as gym
import torch
import json

from sac.agent import SAC
from sac.envs import *
from tqdm import tqdm


def main(args):
    """
    Main function to run the training.
    """
    # Load hyperparameters from config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Post-process config to handle string-encoded lists
    for net in ["q_net", "policy_net"]:
        if net in config and isinstance(config[net].get("hidden_sizes"), str):
            config[net]["hidden_sizes"] = json.loads(config[net]["hidden_sizes"])

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
        env = gym.make(env_name, max_episode_steps=config["train"]["max_episode_steps"])

    # Initialize agent
    agent = SAC(env, config)

    print("Agent initialized. Starting training...")
    metrics = agent.run_training_loop(num_episodes=config["train"]["num_episodes"])

    print(f"Final average return: {metrics['final_avg_return']}")
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example_config_env.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
