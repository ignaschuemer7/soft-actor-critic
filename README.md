# Soft Actor-Critic (SAC)

<p align="center">
  <strong>A modular and extensible implementation of the <a href="https://arxiv.org/abs/1801.01290">Soft Actor-Critic (SAC)</a> algorithm in <a href="https://pytorch.org/">PyTorch</a>.
  <br>
  This repository provides a robust framework for continuous control tasks, featuring hyperparameter tuning, comprehensive logging, and support for a wide range of Gymnasium and custom environments.</strong>
</p>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium 0.29+](https://img.shields.io/badge/gymnasium-0.29%2B-00ADD8.svg)](https://gymnasium.farama.org/)

## Key Features

- **Soft Actor-Critic (SAC) Algorithm:** A full implementation of the SAC algorithm, including the twin Q-network architecture and the automated entropy tuning.
- **Hyperparameter Tuning with Optuna:** The repository includes a script for hyperparameter optimization using [Optuna](https://optuna.org/). The search space can be easily configured using YAML files.
- **Logging System:** The logging system uses [TensorBoard](https://www.tensorflow.org/tensorboard) to log metrics, hyperparameters, and experiment results. It also saves Matplotlib graphs of the episode rewards, lengths, and Q-values.
- **Configuration via YAML Files:** All the hyperparameters for the agent and the training process can be easily configured using YAML files.
- **Jupyter Notebooks for Experimentation:** The repository includes several Jupyter notebooks that demonstrate how to use the SAC agent in different environments.
- **Custom Environments:** The project includes several custom environments that can be used to test and debug the agent.

### Tested Environments:
- **[Gymnasium](https://gymnasium.farama.org/index.html) Benchmarks:**
  - [`BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/): Teach a four-legged walker to traverse uneven terrain efficiently without falling.
    <p align="center">
      <img src="assets/bipedal_walker.gif" alt="BipedalWalker-v3 Agent in action" width="400">
      <br/>
      <em>Our SAC agent trained for 400 episodes (max 1,600 steps each) demonstrating stable locomotion on BipedalWalker-v3.</em>
    </p>
  - [`InvertedPendulum-v5`](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/): Keep a hinged pole balanced upright by moving the cart beneath it.

- **[Donkey Gym](https://github.com/tawnkramer/gym-donkeycar) (OpenAI Gym wrapper for the [Self Driving Sandbox Donkey Simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/)):**
    
  - `DonkeyVae-v0-level-0` (“generated roads”): Maintain the autonomous vehicle on procedurally generated tracks while completing laps quickly.
  - `DonkeyVae-v0-level-2` (“Sparkfun AVC”): Navigate the Sparkfun AVC circuit with minimal lateral error at racing speed.
    <p align="center">
      <img src="assets/donkey_car.gif" alt="DonkeyCar Agent in action" width="400">
      <br/>
      <em>Our SAC agent on DonkeyVae-v0-level-0 after 100 training episodes (max 1,000 steps each) using a 32-D VAE latent observation.</em>
    </p>

    This environment utilizes a Variational Auto-Encoder (VAE) as a feature extractor. The VAE compresses high-dimensional image observations into a lower-dimensional latent space that the SAC agent can effectively use for decision-making.
    
    The VAE implementation, simulator wrapper, and communication code are adapted from [Learning to Drive Smoothly in Minutes](https://github.com/araffin/learning-to-drive-in-5-minutes).
    
- **Custom Gymnasium Environments:**
  - `OneDPointMassReachEnv`: A simple 1D point mass environment.
  - `QuadraticActionRewardEnv`: An environment with a quadratic penalty on actions.
  - `RandomObsBinaryRewardEnv`: An environment with random observations and binary rewards.
  - `ConstantRewardEnv`: An environment that returns a constant reward, useful for debugging.


## Getting Started

### 1. Installation
First, clone the repository and set up your virtual environment:
```bash
git clone https://github.com/ignaschuemer7/RL-SAC.git
cd RL-SAC
python -m venv sac_env
source sac_env/bin/activate
pip install -r requirements.txt
```

### 2. Training the Agent
Train the SAC agent using a configuration file. An example is provided in `configs/example_config_env.yaml`.
```bash
python main.py --config configs/example_config_env.yaml
```
You can create custom YAML files to train on different environments or with different hyperparameters.

### 3. Hyperparameter Search
Optimize hyperparameters with Optuna by running:
```bash
python hparam_search/scripts/run_search.py --search-config hparam_search/configs/search_space.yaml --base-config hparam_search/configs/base_hparams.yaml --n-trials 10 --study-name my-study
```
Customize the search space and base hyperparameters in the respective YAML configuration files.

### 4. Experiment with Jupyter Notebooks
The `notebooks` directory contains examples of how to use the SAC agent in various environments.

### 5. Viewing Logs
Monitor training and hyperparameter search results with TensorBoard. Logs are organized as follows:
- **`runs/`**: For training runs from `main.py`.
- **`hparam_search/hparam_runs/`**: For Optuna hyperparameter searches.
- **`notebooks/runs/`**: For trainings initiated from Jupyter notebooks.

Launch TensorBoard from the project root:
```bash
# For standard training logs
tensorboard --logdir runs/

# For hyperparameter search logs
tensorboard --logdir hparam_search/hparam_runs/

# For notebook-specific logs
tensorboard --logdir notebooks/runs/
```
Access the dashboard at `http://localhost:6006/`.


## Authors

This project was developed as the final project for the **Reinforcement Learning (I404)** course at [Universidad de San Andrés](https://www.udesa.edu.ar/), Argentina, by:

- **Fausto Pettinari** (fpettinari@udesa.edu.ar)  
- **Ignacio Schuemer** (ischuemer@udesa.edu.ar)
- **Santiago Tomas Torres** (storres@udesa.edu.ar)