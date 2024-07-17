from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class LRScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float = 0,
        end_warmup_step: int | None = None,
        end_decay_step: int | None = None,
        end_cooldown_step: int | None = None,
        torch_scheduler: str | None = None,
        torch_scheduler_args: dict | None = None,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        # TODO: use n_steps instead of additive points here to simplify
        self.end_warmup_step = (end_warmup_step - 1) if isinstance(end_warmup_step, int) else end_warmup_step
        self.end_decay_step = (end_decay_step - 1) if isinstance(end_decay_step, int) else end_decay_step
        self.end_cooldown_step = (end_cooldown_step - 1) if isinstance(end_cooldown_step, int) else end_cooldown_step
        self.torch_scheduler = torch_scheduler
        self.torch_scheduler_args = torch_scheduler_args

        if self.end_warmup_step is not None:
            self.warmup_slope = self.max_lr / (self.end_warmup_step)

        self.cooldown_slope: float | None = None

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

    def step(
        self,
        steps: int,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        # Warmup
        if self.end_warmup_step is not None and steps <= self.end_warmup_step:
            if steps == 0:
                self._edit_lr_mult(optimizer, 0)
            else:
                self._edit_lr_add(optimizer, self.warmup_slope)
        # Main
        elif steps != 0 and (self.end_decay_step is None or (self.end_decay_step and steps <= self.end_decay_step)):
            if scheduler is not None:
                scheduler.step()
                if self._get_min_lr_from_optim(optimizer=optimizer) < self.min_lr:
                    self._edit_lr_mult(optimizer, 1)
            else:
                return
        # Cooldown
        elif self.end_cooldown_step is not None and steps <= self.end_cooldown_step:
            if self.cooldown_slope is None:
                self.cooldown_slope = (self.min_lr - self._get_min_lr_from_optim(optimizer)) / (
                    self.end_cooldown_step - steps + 1
                )
            self._edit_lr_add(optimizer, self.cooldown_slope)
