import torch
from typing import Any, Dict
from datetime import datetime

activation_lookup = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "gelu": torch.nn.GELU,
    "selu": torch.nn.SELU,
    "identity": torch.nn.Identity,
}


def get_sb3_sac_params(
    env, config: Dict[str, Any], seed: int, env_id: str = ""
) -> Dict[str, Any]:
    """Map custom SAC config to Stable Baselines3 SAC parameters."""

    sb3_activation = activation_lookup.get(
        config["policy_net"].get("hidden_layers_act", "relu"), torch.nn.ReLU
    )

    sb3_policy_kwargs = {
        "net_arch": {
            "pi": config["policy_net"]["hidden_sizes"],
            "qf": config["q_net"]["hidden_sizes"],
        },
        "activation_fn": sb3_activation,
    }

    sb3_env = env

    timestamp = datetime.now().strftime(config["logger"]["timestamp_format"])
    sb3_tensorboard_log = f"{config['logger'].get('log_dir', 'runs')}/{env_id}/{config['logger']['agent_name']}_sb3/{config['logger']['run_name']}-{timestamp}"

    sb3_params = {
        "policy": "MlpPolicy",
        "env": sb3_env,
        "learning_rate": config["sac"]["actor_lr"],  # SB3 uses one LR for all nets
        "buffer_size": config["buffer"]["capacity"],
        "learning_starts": config["train"]["warming_steps"],
        "batch_size": config["train"]["batch_size"],
        "tau": config["sac"]["tau"],
        "gamma": config["sac"]["gamma"],
        "train_freq": (1, "step"),
        "gradient_steps": config["train"]["gradient_steps_per_update"],
        "ent_coef": (
            "auto" if config["sac"]["auto_entropy_tuning"] else config["sac"]["alpha"]
        ),
        "target_entropy": -sb3_env.action_space.shape[0],
        "policy_kwargs": sb3_policy_kwargs,
        "device": config["train"]["device"],
        "seed": seed,
        # "verbose": 1,
        # "tensorboard_log": sb3_tensorboard_log,
    }

    return sb3_params
