from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use((Path(__file__).parent / "custom.mplstyle").resolve())

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(
        self,
        cfg: Dict[str, Any],
        run_name: Optional[str] = None,
        env_name: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        self.cfg = cfg
        self.env_name = env_name or cfg["env_name"] or "Environment"
        self.agent_name = agent_name or cfg["agent_name"] or "Agent"
        base_name = run_name or cfg["run_name"] or "sac"
        if cfg["use_timestamp"]:
            base_name = (
                f"{base_name}-{datetime.now().strftime(cfg['timestamp_format'])}"
            )

        self.run_id = base_name
        self.run_dir = (
            Path(cfg["log_dir"]) / self.env_name / self.agent_name / self.run_id
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_writer = SummaryWriter(
            self.run_dir.as_posix(),
            flush_secs=cfg["flush_secs"],
            filename_suffix="_metrics",
        )
        self.hparams_writer = SummaryWriter(
            self.run_dir.as_posix(),
            flush_secs=cfg["flush_secs"],
            filename_suffix="_hparams",
        )
        self._hparams_logged = False
        self.episode_rewards = []
        self.episode_lengths = []
        self.q1_values = []
        self.q2_values = []

    def log_episode_metrics(self, episode_idx: int, reward: float, length: int) -> None:
        if not self.cfg["log_episode_stats"]:
            return
        self.metrics_writer.add_scalar("Episode/Reward", reward, episode_idx)
        self.metrics_writer.add_scalar("Episode/Length", length, episode_idx)
        # Save for matplotlib graphs:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

    def save_matplotlib_graphs(self) -> None:
        if self.episode_rewards:
            self.make_and_save_graph(
                1,
                [self.episode_rewards],
                f"Episode Rewards Over Time - {self.env_name} - {self.agent_name}",
                "Episode",
                "Reward",
                f"episode_rewards-{self.env_name}_{self.agent_name}.pdf",
            )
        if self.episode_lengths:
            self.make_and_save_graph(
                1,
                [self.episode_lengths],
                f"Episode Lengths Over Time - {self.env_name} - {self.agent_name}",
                "Episode",
                "Length",
                f"episode_lengths-{self.env_name}_{self.agent_name}.pdf",
            )
        if self.q1_values and self.q2_values:
            self.make_and_save_graph(
                2,
                [self.q1_values, self.q2_values],
                f"Q-Values Over Time - {self.env_name} - {self.agent_name}",
                "Step",
                "Q-Value",
                f"q_values-{self.env_name}_{self.agent_name}.pdf",
                legend=["Q1", "Q2"],
            )

    def log_q_values(self, q1_value: float, q2_value: float, step: int) -> None:
        if not self.cfg["log_q_values"]:
            return
        self.metrics_writer.add_scalar("QValues/Q1", q1_value, step)
        self.metrics_writer.add_scalar("QValues/Q2", q2_value, step)
        self.q1_values.append(q1_value)
        self.q2_values.append(q2_value)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        if self._hparams_logged:
            return
        prepared_hparams = self._prepare_hparams(hparams)
        prepared_metrics = {k: float(v) for k, v in metrics.items()}
        if not prepared_metrics:
            prepared_metrics = {"placeholder_metric": 0.0}
        self.hparams_writer.add_hparams(prepared_hparams, prepared_metrics)
        self._hparams_logged = True

    def flush(self) -> None:
        self.metrics_writer.flush()
        self.hparams_writer.flush()

    def close(self) -> None:
        self.flush()
        self.metrics_writer.close()
        self.hparams_writer.close()

    def save(self) -> None:
        # save rewards and lengths to a file
        rewards_path = self.run_dir / "episode_rewards.txt"
        lengths_path = self.run_dir / "episode_lengths.txt"
        with rewards_path.open("w") as f:
            for reward in self.episode_rewards:
                f.write(f"{reward}\n")
        with lengths_path.open("w") as f:
            for length in self.episode_lengths:
                f.write(f"{length}\n")

    def load(self, rewards_path: str = None, lengths_path: str = None) -> None:
        if rewards_path is None:
            rewards_path = self.run_dir / "episode_rewards.txt"
        if lengths_path is None:
            lengths_path = self.run_dir / "episode_lengths.txt"
        with open(rewards_path, "r") as f:
            self.episode_rewards = [float(line.strip()) for line in f]
        with open(lengths_path, "r") as f:
            self.episode_lengths = [int(line.strip()) for line in f]

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def make_and_save_graph(
        self,
        number_of_curves: int,
        data: list,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        legend: list[str] = None,
    ) -> None:
        plt.figure()
        for i in range(number_of_curves):
            plt.plot(data[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.grid(True)
        graph_path = self.run_dir / filename
        if legend:
            plt.legend(legend)
        plt.savefig(graph_path)
        plt.close()

    @staticmethod
    def _prepare_hparams(hparams: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}

        def _flatten(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, val in value.items():
                    child_key = f"{prefix}/{key}" if prefix else key
                    _flatten(child_key, val)
            else:
                flat[prefix] = value

        _flatten("", hparams)
        sanitized: Dict[str, Any] = {}
        for key, value in flat.items():
            if isinstance(value, (int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
