"""Experiment logging abstraction for DFLIP.

This module provides a unified logging interface that can be backed by
- Weights & Biases (wandb)
- SwanLab
- or a no-op implementation (disabled logging).

Training code should only depend on :class:`BaseLogger` / ``create_logger``
so that we can switch backends via config without touching the core logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

# Optional dependencies: wandb & swanlab
try:  # noqa: SIM105
    import wandb  # type: ignore
except ImportError:  # noqa: F401
    wandb = None  # type: ignore

try:  # noqa: SIM105
    import swanlab  # type: ignore
except ImportError:  # noqa: F401
    swanlab = None  # type: ignore


class BaseLogger(ABC):
    """Abstract experiment logger interface.

    Minimal interface for scalar logging. If in future you need to log
    images / tables / models, consider extending this with more methods
    and implement them in each backend.
    """

    @abstractmethod
    def log(self, data: Dict[str, Any]) -> None:
        """Log a flat dict of scalars/serializable values."""

    @abstractmethod
    def is_active(self) -> bool:
        """Whether the underlying logger is actually recording anything."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize the run (if necessary)."""


class NoOpLogger(BaseLogger):
    """Logger that does nothing.

    Useful when logging is disabled or dependencies are missing.
    """

    def log(self, data: Dict[str, Any]) -> None:  # noqa: ARG002
        return

    def is_active(self) -> bool:
        return False

    def finish(self) -> None:
        return


class WandbLogger(BaseLogger):
    """Weights & Biases backend implementation."""

    def __init__(self, config: Dict[str, Any], logging_cfg: Dict[str, Any]) -> None:
        self._active = False
        self._run = None

        if wandb is None:
            print("wandb 未安装，已禁用 Weights & Biases 记录功能。")
            return

        # backend 显式为 wandb，或者兼容旧配置 use_wandb=True
        backend = logging_cfg.get("backend")
        use_wandb = logging_cfg.get("use_wandb", False)
        if backend not in (None, "wandb") and not use_wandb:
            # 显式指定了其他 backend
            return

        project = logging_cfg.get("wandb_project", "dflip")
        entity = logging_cfg.get("wandb_entity")
        run_name = logging_cfg.get("wandb_run_name")

        self._run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=run_name,
        )
        self._active = self._run is not None

    def log(self, data: Dict[str, Any]) -> None:
        if self._active:
            wandb.log(data)

    def is_active(self) -> bool:
        return self._active

    def finish(self) -> None:
        if self._active:
            wandb.finish()


class SwanLabLogger(BaseLogger):
    """SwanLab backend implementation."""

    def __init__(self, config: Dict[str, Any], logging_cfg: Dict[str, Any]) -> None:
        self._active = False
        self._run = None

        if swanlab is None:
            print("swanlab 未安装，已禁用 SwanLab 记录功能。")
            return

        backend = logging_cfg.get("backend")
        if backend != "swanlab":
            return

        project = logging_cfg.get("swanlab_project", "dflip")
        experiment_name = logging_cfg.get("swanlab_experiment_name")

        self._run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config=config,
        )
        # 一般 swanlab.init 没有返回值，这里主要是做个布尔标记
        self._active = True

    def log(self, data: Dict[str, Any]) -> None:
        if self._active:
            swanlab.log(data)

    def is_active(self) -> bool:
        return self._active

    def finish(self) -> None:
        if self._active:
            swanlab.finish()


def create_logger(config: Dict[str, Any]) -> BaseLogger:
    """Factory function to create an experiment logger from config.

    Config example::

        logging:
          backend: "swanlab"   # "swanlab" | "wandb" | "none"
          use_wandb: false      # for backward compatibility
          wandb_project: "dflip-linguistic-profiling"
          wandb_entity: null
          wandb_run_name: null
          swanlab_project: "dflip"
          swanlab_experiment_name: null
    """

    logging_cfg: Dict[str, Any] = config.get("logging", {}) or {}
    backend = logging_cfg.get("backend")

    # 兼容旧配置：如果没写 backend 但 use_wandb=True，默认用 wandb
    if backend is None and logging_cfg.get("use_wandb", False):
        backend = "wandb"

    backend = (backend or "none").lower()

    if backend == "wandb":
        logger = WandbLogger(config, logging_cfg)
        if logger.is_active():
            return logger
        return NoOpLogger()

    if backend == "swanlab":
        logger = SwanLabLogger(config, logging_cfg)
        if logger.is_active():
            return logger
        return NoOpLogger()

    # backend == "none" 或未识别的情况
    return NoOpLogger()
