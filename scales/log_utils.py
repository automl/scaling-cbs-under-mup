from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List
from warnings import warn

import lightning as L
import pandas as pd
import torch
import torch.nn
from tbparse import SummaryReader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from scales.config import TrainConfig
from scales.utils import gradient_l2_norm_per_layer, total_gradient_l2_norm

MODEL_HPARAMS = ["block_size", "batch_size", "d_model", "n_layer", "n_head", "head_size"]


def load_tb(output_dir: str | Path, train_config_file_name: str | None = "train_config_post_init") -> pd.DataFrame:
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    reader = SummaryReader(str(log_dir))
    df = pd.pivot_table(reader.scalars, values="value", columns="tag", index="step").ffill()

    if train_config_file_name is None:
        return df

    # Get HPs from written config file
    train_config_path = output_dir / f"{train_config_file_name}.yaml"
    train_config = TrainConfig.from_path(train_config_path)
    model_config = train_config.model_config

    for hp_name in MODEL_HPARAMS:
        if hp_name == "batch_size":
            df[hp_name] = train_config.accumulation_iters * train_config.micro_batch_size
        else:
            df[hp_name] = getattr(model_config, hp_name, None)

    return df


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

    global_log_step: int = 1
    tracked_metrics: dict[str, int] = field(default_factory=dict)
    log_dir: str | Path | None = None
    suppress_all_logs: bool = False

    def __post_init__(self) -> None:
        """Function to be called after log_dir change."""
        self.writer = None
        if self.tracked_metrics and self.log_dir:
            # Warn for typos
            _ = [self.get_metric(metric) for metric in self.tracked_metrics]
        # resolve logging frequency
        for k, v in self.tracked_metrics.items():
            if v == 0 or v is None:
                self.tracked_metrics[k] = self.global_log_step

    def start_logger(self) -> None:
        if self.tracked_metrics and self.log_dir:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.total_logits_mean = 0
            self.max_attn_logit = None
            self.max_attn_logits_per_layer: list[float] = []
            self.total_logits_max = None

    def log_check(self, func: Callable, step: int, last: bool = False) -> bool:
        if self.suppress_all_logs:
            return False
        # Ignore step count
        if last:
            return func.__name__ in self.tracked_metrics
        # sets the boolean flag if the metric is in the tracked metrics or to global log step
        _step = self.tracked_metrics.get(func.__name__, self.global_log_step)
        return step % _step == 0 and func.__name__ in self.tracked_metrics

    def get_metric(self, metric: str) -> Any:
        try:
            return getattr(self, metric)
        except AttributeError:
            warn(f"Metric: {metric} is not defined in {self.__module__}.{self.__class__}, please check the spelling")

    @should_log
    def learning_rate(self, optimizer: Optimizer, step: int) -> None:
        self.writer.add_scalar(tag="Learning Rate", scalar_value=optimizer.param_groups[-1]["lr"], global_step=step)

    @should_log
    def output_logits_max(
        self, logits: torch.Tensor, step: int, fabric: L.Fabric, is_accumulating: bool, accumulation_iters: int
    ) -> None:
        logits_max = logits.max().item()
        if self.total_logits_max is not None:
            self.total_logits_max = max(self.total_logits_max, logits_max)
        else:
            self.total_logits_max = logits_max
        if not is_accumulating:
            self.total_logits_max = fabric.all_reduce(torch.tensor(self.total_logits_max), reduce_op="max")
            self.writer.add_scalar(tag="Output Logits/Max", scalar_value=self.total_logits_max, global_step=step)
            self.total_logits_max = None

    @should_log
    def output_logits_mean(
        self, logits: torch.Tensor, step: int, fabric: L.Fabric, is_accumulating: bool, accumulation_iters: int
    ) -> None:
        logits_mean = logits.mean().item()
        self.total_logits_mean += logits_mean / accumulation_iters
        if not is_accumulating:
            self.total_logits_mean = fabric.all_reduce(torch.tensor(self.total_logits_mean), reduce_op="mean")
            self.writer.add_scalar(tag="Output Logits/Mean", scalar_value=self.total_logits_mean, global_step=step)
            self.total_logits_mean = 0

    @should_log
    def max_attention_logits_per_layer(
        self, attn_logits: List, step: int, fabric: L.Fabric, is_accumulating: bool
    ) -> None:
        if not attn_logits:
            raise ValueError("Error happened during sharing the attn_logits from the model")
        if not self.max_attn_logits_per_layer:
            self.max_attn_logits_per_layer = attn_logits
        else:
            self.max_attn_logits_per_layer = [max(a, b) for a, b in zip(self.max_attn_logits_per_layer, attn_logits)]
        if not is_accumulating:
            self.max_attn_logits_per_layer = fabric.all_reduce(self.max_attn_logits_per_layer, reduce_op="max")
            for i, logit in enumerate(self.max_attn_logits_per_layer):
                self.writer.add_scalar(tag=f"Max Attention Logits/Layer{i}", scalar_value=logit, global_step=step)
            self.max_attn_logits_per_layer = []

    @should_log
    def max_attention_logits_all(self, attn_logits: List, step: int, fabric: L.Fabric, is_accumulating: bool) -> None:
        if not attn_logits:
            raise ValueError("Error happened during sharing the attn_logits from the model")
        if self.max_attn_logit is None:
            self.max_attn_logit = max(attn_logits)
        else:
            max_between_logits = max(attn_logits)
            self.max_attn_logit = max(max_between_logits, self.max_attn_logit)
        if not is_accumulating:
            self.max_attn_logit = fabric.all_reduce(self.max_attn_logit, reduce_op="max")
            self.writer.add_scalar(tag="Max Attention Logits/All", scalar_value=self.max_attn_logit, global_step=step)
            self.max_attn_logit = None

    @should_log
    def total_gradient_norm(self, model: torch.nn.Module, step: int) -> None:
        total_norm = total_gradient_l2_norm(model)
        self.writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=step)

    @should_log
    def gradient_norm_per_layer(self, model: torch.nn.Module, step: int) -> None:
        layer_grad_norms = gradient_l2_norm_per_layer(model, step)

        # log to TensorBoard as separate plots
        for layer, norm in layer_grad_norms.items():
            self.writer.add_scalar(tag=f"Per-layer Gradient Norm/layer{layer}", scalar_value=norm, global_step=step)

    @should_log
    def train_loss(self, loss: float, step: int) -> None:
        self.writer.add_scalar(tag="Train Loss", scalar_value=loss, global_step=step)

    @should_log
    def validation_loss(self, value: float, step: int) -> None:
        self.writer.add_scalar(tag="Validation Loss", scalar_value=value, global_step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
