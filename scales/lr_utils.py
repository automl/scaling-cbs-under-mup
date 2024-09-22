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
        torch_scheduler: str | None = None,
        torch_scheduler_args: dict | None = None,
        warmup_frac: float | None = None,
        cool_down_frac: float | None = None,
        cooldown_type: str = "linear",
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.n_warmup = int(max_steps * warmup_frac) if warmup_frac is not None else 0
        self.n_cooldown = int(max_steps * cool_down_frac) if cool_down_frac is not None else 0
        self.n_middle = max_steps - self.n_warmup - self.n_cooldown

        self.end_cooldown_step = self.n_warmup + self.n_middle + self.n_cooldown
        self.end_warmup_step = self.n_warmup
        self.end_decay_step = self.n_warmup + self.n_middle

        # for initial optimizer step==0
        self.init_lr = 0.0 if self.n_warmup else max_lr

        # TODO: write an adapter for all torch.optim.LRScheduler classes
        if torch_scheduler and torch_scheduler == "CosineAnnealingLR":
            torch_scheduler_args = {} if torch_scheduler_args is None else torch_scheduler_args
            torch_scheduler_args["T_max"] = max_steps - self.n_warmup - self.n_cooldown

        self.torch_scheduler = torch_scheduler
        self.torch_scheduler_args = torch_scheduler_args
        self.cooldown_type = cooldown_type

        if self.end_warmup_step:
            self.warmup_slope = self.max_lr / self.end_warmup_step


        self.cooldown_slope: float | None = None
        self.min_lr_at_cooldown_start: float | None = None

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def _instantiate_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if isinstance(self.torch_scheduler, str):
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.torch_scheduler)
            return scheduler_cls(optimizer, **self.torch_scheduler_args)
        raise ValueError("The `lr_scheduler` argument should be a string")

    def _get_min_lr_from_optim(self, optimizer: Optimizer) -> float:
        min_lr = 1
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            if lr < min_lr:
                min_lr = lr
        return min_lr

    def _edit_lr_add(self, optimizer: Optimizer, scale: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] += scale

    def _edit_lr_mult(self, optimizer: Optimizer, scale: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= scale

    def cooldown_step(self, optimizer: Optimizer, step: int) -> None:
        if self.min_lr_at_cooldown_start is None:
            self.min_lr_at_cooldown_start = self._get_min_lr_from_optim(optimizer)
        if self.cooldown_type == "linear":
            if self.cooldown_slope is None:
                self.cooldown_slope = (self.min_lr - self.min_lr_at_cooldown_start) / (self.end_cooldown_step - step)
            self._edit_lr_add(optimizer, self.cooldown_slope)
        elif self.cooldown_type == "rsqrt":
            previous_min_lr = self._get_min_lr_from_optim(optimizer)

            # max_lr * (1 - sqrt((step - cooldown_start)/n_decay)) - previous_lr
            additive = (
                self.min_lr_at_cooldown_start * (1 - math.sqrt((step - self.end_decay_step) / self.n_cooldown))
                - previous_min_lr
            )
            self._edit_lr_add(optimizer, additive)

    def step(self, steps: int, optimizer: Optimizer) -> None:
        # Warmup
        if steps <= self.end_warmup_step:
            if steps == 0:
                self._edit_lr_mult(optimizer, 0)
            else:
                self._edit_lr_add(optimizer, self.warmup_slope)
        # Main
        elif steps != 0 and self.end_decay_step and steps <= self.end_decay_step:
            if self.torch_scheduler is not None:
                if self.scheduler is None:
                    self.scheduler = self._instantiate_lr_scheduler(optimizer)
                
                if self._get_min_lr_from_optim(optimizer=optimizer) > self.min_lr:
                    self.scheduler.step()
                    # self._edit_lr_mult(optimizer, 1)
            else:
                return
        # Cooldown
        elif steps <= self.end_cooldown_step:
            self.cooldown_step(optimizer, steps)
