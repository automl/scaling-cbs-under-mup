from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class LRScheduler:
    def __init__(
        self,
        max_lr: float,
        max_steps: int,
        min_lr: float = 0,
        cosine_scheduler: bool = False,
        scheduler_args: dict | None = None,
        warmup_frac: float | None = None,
        cool_down_frac: float | None = None,
        cooldown_type: str = "linear",
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.n_warmup = int(max_steps * warmup_frac) if warmup_frac is not None else 0
        self.n_cooldown = int(max_steps * cool_down_frac) if cool_down_frac is not None else 0
        self.n_middle = max_steps - self.n_warmup - self.n_cooldown

        self.end_cooldown_step = self.n_warmup + self.n_middle + self.n_cooldown - 1
        self.end_warmup_step = self.n_warmup - 1
        self.end_decay_step = self.n_warmup + self.n_middle - 1
        self.max_lr_final_decay: float | None = None

        if cosine_scheduler:
            scheduler_args = {} if scheduler_args is None else scheduler_args
            scheduler_args["T_max"] = max_steps - self.n_warmup - self.n_cooldown
            self.max_lr_final_decay = (
                scheduler_args.pop("max_lr_final_decay") if scheduler_args.get("max_lr_final_decay") else 0
            )

        self.cosine_scheduler = cosine_scheduler
        self.scheduler_args = scheduler_args
        self.cooldown_type = cooldown_type

        self.learning_rates: list[float] = []

        self.cooldown_slope: list[float] = []
        self.warmup_slope: list[float] = []
        self.min_lr_at_cooldown_start: list[float] = []

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def setup_scheduler(self, optimizer: Optimizer) -> None:
        for _, param_group in enumerate(optimizer.param_groups):
            self.learning_rates.append(param_group["lr"])

        if self.n_warmup != 0 and self.end_warmup_step != 0:
            for _, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = 0
            for learning_rate in self.learning_rates:
                self.warmup_slope.append(learning_rate / self.end_warmup_step)

    def _instantiate_lr_scheduler(self, optimizer: Optimizer) -> None:
        lr_lambdas = []

        for lr in self.learning_rates:
            lr_lambdas.append(
                lambda step, lr=lr: (
                    lr * self.max_lr_final_decay
                    + (lr - lr * self.max_lr_final_decay)
                    * (1 + math.cos(math.pi * step / self.scheduler_args["T_max"]))
                    / 2
                )
                / lr
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambdas,
        )

    def _get_min_lr_from_optim(self, optimizer: Optimizer) -> None:
        min_lr = 1
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            if lr < min_lr:
                self.min_lr_at_cooldown_start.append(lr)

    def cooldown_step(self, optimizer: Optimizer, step: int) -> None:
        if len(self.min_lr_at_cooldown_start) == 0:
            self._get_min_lr_from_optim(optimizer)
        if len(self.cooldown_slope) == 0:
            for min_lr in self.min_lr_at_cooldown_start:
                self.cooldown_slope.append((self.min_lr - min_lr) / (self.end_cooldown_step - step + 1))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] += self.cooldown_slope[i]

    def step(self, steps: int, optimizer: Optimizer) -> None:
        # Warmup
        if steps <= self.end_warmup_step and self.end_warmup_step != 0:
            if steps != 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] += self.warmup_slope[i]
        # Main
        elif steps != 0 and self.end_decay_step and steps <= self.end_decay_step:
            if self.cosine_scheduler:
                if self.scheduler is None:
                    self._instantiate_lr_scheduler(optimizer)
                if self.scheduler:
                    self.scheduler.step()
                else:
                    raise ValueError("The cosine scheduler was not defined properly")
            else:
                return
        # Cooldown
        elif steps <= self.end_cooldown_step:
            self.cooldown_step(optimizer, steps)
