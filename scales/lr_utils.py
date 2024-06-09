from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class LRScheduler:
    def __init__(
        self,
        init_lr: float,
        min_lr: float = 0,
        end_warmup_step: int | None = None,
        end_decay_step: int | None = None,
        end_cooldown_step: int | None = None,
        torch_scheduler: str | None = None,
        torch_scheduler_args: dict | None = None,
    ) -> None:
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.end_warmup_step = (end_warmup_step - 1) if isinstance(end_warmup_step, int) else end_warmup_step
        self.end_decay_step = (end_decay_step - 1) if isinstance(end_decay_step, int) else end_decay_step
        self.end_cooldown_step = (end_cooldown_step - 1) if isinstance(end_cooldown_step, int) else end_cooldown_step
        self.torch_scheduler = torch_scheduler
        self.torch_scheduler_args = torch_scheduler_args

        self._inital_cooldown_lr: float
        self._cooldown_shift: int = 0
        self._first_cooldown_step: bool = True

    def _instantiate_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if isinstance(self.torch_scheduler, str):
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.torch_scheduler)
            return scheduler_cls(optimizer, **self.torch_scheduler_args)
        raise ValueError("The `lr_scheduler` argument should be a string")

    def _get_lr_from_optim(self, optimizer: Optimizer) -> float:
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        return lr

    def _edit_lr(self, optimizer: Optimizer, new_lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _get_cooldown_shift(self) -> int:
        if self.end_decay_step:
            return self.end_decay_step
        if self.end_warmup_step and self.end_decay_step is None:
            return self.end_warmup_step
        return 0

    def step(
        self, steps: int, optimizer: Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    ) -> None:
        if self.end_warmup_step is not None and steps <= self.end_warmup_step:
            new_lr = self.init_lr * steps / (self.end_warmup_step)
            self._edit_lr(optimizer, new_lr)
        elif steps != 0 and (self.end_decay_step is None or (self.end_decay_step and steps <= self.end_decay_step)):
            if scheduler is not None:
                scheduler.step()
                if self._get_lr_from_optim(optimizer=optimizer) < self.min_lr:
                    self._edit_lr(optimizer, self.min_lr)
            else:
                return
        elif self.end_cooldown_step and steps <= self.end_cooldown_step:
            if self._first_cooldown_step is True:
                self._inital_cooldown_lr = self._get_lr_from_optim(optimizer)
                self._cooldown_shift = self._get_cooldown_shift()
                self._first_cooldown_step = False
            new_lr = self._inital_cooldown_lr - (
                (self._inital_cooldown_lr - self.min_lr) * (steps - self._cooldown_shift)
            ) / (self.end_cooldown_step - self._cooldown_shift)
            self._edit_lr(optimizer, new_lr)
