import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

#  SAVE FUNCTIONS
def save_rewards(run_dir: str, rewards: List[float]) -> None:
    """
    Saves the list of episode rewards to episode_rewards.npy
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "episode_rewards.npy", np.array(rewards, dtype=np.float32))

def save_lengths(run_dir: str, lengths: List[int]) -> None:
    """
    Saves the list of episode lengths to episode_lengths.npy
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "episode_lengths.npy", np.array(lengths, dtype=np.int32))

#  LOAD FUNCTIONS
def load_rewards(run_dir: str) -> List[float]:
    """
    Loads episode_rewards.npy and returns the rewards list
    """
    run_dir = Path(run_dir)
    return np.load(run_dir / "episode_rewards.npy").astype(float).tolist()

def load_lengths(run_dir: str) -> List[int]:
    """
    Loads episode_lengths.npy and returns the lengths list
    """
    run_dir = Path(run_dir)
    return np.load(run_dir / "episode_lengths.npy").astype(int).tolist()

# Make graph function
def make_and_save_graph(
        number_of_curves: int,
        data: list,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        run_dir: Path,
        legend: list[str] = None,
    ) -> None:
        plt.figure()
        for i in range(number_of_curves):
            plt.plot(data[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(True)
        graph_path = run_dir + "/" + filename
        if legend:
            plt.legend(legend)
        plt.savefig(graph_path)
        plt.close()