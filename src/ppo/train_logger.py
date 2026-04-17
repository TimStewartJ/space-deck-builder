"""Training-specific logging and metrics recording.

Provides structured logging (Python ``logging`` module) and a JSONL metrics
writer for machine-parseable per-update training data.  The game-engine
logger (``src.utils.logger``) is intentionally left untouched — it serves a
different purpose and is disabled during training anyway.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def setup_training_logger(
    log_dir: str = "logs",
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Create (or reconfigure) the ``training`` logger.

    Returns a logger with two handlers:

    * **Console** — concise ``[HH:MM:SS] message`` at *console_level*.
    * **File** — full ``[YYYY-MM-DD HH:MM:SS] LEVEL message`` at *file_level*,
      written to ``<log_dir>/training_<timestamp>.log``.

    Calling this multiple times in the same process is safe: existing
    handlers are removed before new ones are added.
    """
    logger = logging.getLogger("training")
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    # Remove any existing handlers (idempotent across multiple train() calls)
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)

    # Console handler — compact timestamp, writes to stdout (not stderr)
    # to match the old print()-based output that scripts may pipe/redirect.
    import sys
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler — full timestamp + level
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include PID to prevent file collisions when multiple runs start
    # within the same second (e.g. parallel sweeps or test harnesses).
    pid = os.getpid()
    file_handler = logging.FileHandler(log_path / f"training_{ts}_{pid}.log", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)

    return logger


class MetricsWriter:
    """Append-only JSONL writer for per-update training metrics.

    Each call to :meth:`write` appends a single JSON line to the metrics
    file.  The file is flushed after every write so partial runs are always
    readable.

    Usage::

        writer = MetricsWriter("logs")
        writer.write({"update": 1, "actor_loss": 0.12, ...})
        writer.close()
    """

    def __init__(self, log_dir: str = "logs"):
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self._path = path / f"metrics_{ts}_{pid}.jsonl"
        self._file: TextIO = open(self._path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, metrics: dict[str, Any]) -> None:
        """Append *metrics* as a single JSON line."""
        self._file.write(json.dumps(metrics, default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def format_ppo_metrics(metrics: dict[str, Any]) -> str:
    """Format PPO update metrics into a concise single-line summary.

    Example output::

        actor=-0.012 critic=0.34 ent=3.21 kl=0.008 clip=0.12 gnorm=0.42 ev=0.78
    """
    parts = []
    _add = parts.append

    def _fmt(key: str, label: str, fmt: str = ".3f") -> None:
        val = metrics.get(key)
        if val is not None:
            _add(f"{label}={val:{fmt}}")

    _fmt("actor_loss", "actor")
    _fmt("critic_loss", "critic")
    _fmt("entropy", "ent", ".2f")
    _fmt("approx_kl", "kl", ".4f")
    _fmt("clip_fraction", "clip", ".3f")
    _fmt("total_grad_norm", "gnorm", ".3f")
    _fmt("explained_variance", "ev", ".2f")

    return "  ".join(parts)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for non-standard types."""
    if hasattr(obj, "item"):  # torch.Tensor scalar / numpy scalar
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
