from __future__ import annotations

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
        self.env_name = env_name or cfg['env_name'] or "Environment"
        self.agent_name = agent_name or cfg['agent_name'] or "Agent"
        base_name = run_name or cfg['run_name'] or "sac"
        if cfg['use_timestamp']:
            base_name = f"{base_name}-{datetime.now().strftime(cfg['timestamp_format'])}"

        self.run_id = base_name
        self.run_dir = Path(cfg['log_dir']) / self.env_name / self.agent_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_writer = SummaryWriter(
            self.run_dir.as_posix(),
            flush_secs=cfg['flush_secs'],
            filename_suffix="_metrics",
        )
        self.hparams_writer = SummaryWriter(
            self.run_dir.as_posix(),
            flush_secs=cfg['flush_secs'],
            filename_suffix="_hparams",
        )
        self._hparams_logged = False

    def log_episode_metrics(self, episode_idx: int, reward: float, length: int) -> None:
        if not self.cfg['log_episode_stats']:
            return
        self.metrics_writer.add_scalar("Episode/Reward", reward, episode_idx)
        self.metrics_writer.add_scalar("Episode/Length", length, episode_idx)

    def log_q_values(self, q1_value: float, q2_value: float, step: int) -> None:
        if not self.cfg['log_q_values']:
            return
        self.metrics_writer.add_scalar("QValues/Q1", q1_value, step)
        self.metrics_writer.add_scalar("QValues/Q2", q2_value, step)

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

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

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
