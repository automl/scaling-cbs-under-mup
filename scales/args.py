from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from warnings import warn

import lightning as L
import torch
import torch.nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from scales.utils import total_gradient_norm


def should_log(func: Callable) -> Callable:
    @wraps(func)
    def decorated(self: Any, *args: Any, **kwargs: Any) -> None:
        step = kwargs.get("step", 0)
        last = kwargs.pop("last", False)
        if self.log_check(func=func, step=step, last=last):
            # no grad tracking for logging
            with torch.inference_mode():
                func(self, *args, **kwargs)

    return decorated


@dataclass
class LoggingArgs:
    """Logs every certain number of steps.
    To add a new metric, simply define a method decorated with `should_log`
    which takes in the argument `step` and any other necessary args, kwargs and logs the data into `self.writer`
    Example:
    ```
    @should_log
    def my_new_metric_name(self, *args, step:int, **kwargs) -> None:
         self.writer...
    ```
    Then call this function whenever necessary and if the name of the metric is in the `self.tracked_metrics` list
    the metric will be logged each `log_step` steps. To skip the `log_step` check, pass in extra arg: last=True
    """

    log_step: int = 5
    tracked_metrics: list[str] = field(default_factory=list)
    log_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """Function to be called after log_dir change."""
        if self.tracked_metrics and self.log_dir:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.total_logits_mean = 0
            # Warn for typos
            _ = [self.get_metric(metric) for metric in self.tracked_metrics]
        else:
            self.writer = None

    def update_logdir(self, log_dir: str | Path | None) -> None:
        self.log_dir = log_dir
        self.__post_init__()

    def log_check(self, func: Callable, step: int, last: bool = False) -> bool:
        if last:
            return func.__name__ in self.tracked_metrics
        return step % self.log_step == 0 and func.__name__ in self.tracked_metrics

    def get_metric(self, metric: str) -> Any:
        try:
            return getattr(self, metric)
        except AttributeError:
            warn(f"Metric: {metric} is not defined in {self.__module__}.{self.__class__}, please check the spelling")

    @should_log
    def learning_rate(self, optimizer: Optimizer, step: int) -> None:
        self.writer.add_scalar(tag="Learning Rate", scalar_value=optimizer.param_groups[-1]["lr"], global_step=step)

    @should_log
    def output_logits_mean(
        self, logits: torch.Tensor, step: int, fabric: L.Fabric, is_accumulating: bool, accumulation_iters: int
    ) -> None:
        logits_mean = logits.mean().item()
        self.total_logits_mean += logits_mean / accumulation_iters
        if not is_accumulating:
            self.total_logits_mean = fabric.all_reduce(torch.tensor(self.total_logits_mean), reduce_op="mean")
            self.writer.add_scalar(tag="Output Logits Mean", scalar_value=self.total_logits_mean, global_step=step)
            self.total_logits_mean = 0

    @should_log
    def total_gradient_norm(self, model: torch.nn.Module, step: int) -> None:
        total_norm = total_gradient_norm(model)
        self.writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=step)

    @should_log
    def train_loss(self, loss: float, step: int) -> None:
        self.writer.add_scalar(tag="Train Loss", scalar_value=loss, global_step=step)

    @should_log
    def validation_loss(self, value: float, step: int) -> None:
        self.writer.add_scalar(tag="Validation Loss", scalar_value=value, global_step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
