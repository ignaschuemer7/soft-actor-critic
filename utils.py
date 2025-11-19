from typing import List
import torch.nn as nn
import numpy as np

def build_mlp(
    state_size: int,
    hidden_sizes: List[int],
    action_size: int,
    activation: nn.Module = nn.Tanh,
    output_activation: nn.Module = nn.Identity,
) -> nn.Sequential:
    """
    Build a simple MLP network.
    
    Args:
        state_size: Dimension of state space
        hidden_sizes: List of hidden layer sizes
        action_size: Number of actions
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
    
    Returns:
        nn.Sequential: MLP model
    """
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty")
    
    layer_sizes = [state_size] + hidden_sizes + [action_size]
    
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)

