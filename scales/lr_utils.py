from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod

import torch
from torch.optim.optimizer import Optimizer


class BaseLR(ABC):
    def __init__(self, init_lr: float) -> None:
        assert init_lr > 0, "The learning rate should be positive"
        self.init_lr = init_lr

    @abstractmethod
    def get_lr(self, step: int, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError

    def _get_lr_from_optim(self, optimizer: torch.optim.Optimizer) -> float:
        for param_group in optimizer.param_groups:
            lr = float(param_group["lr"])
        return lr


class ConstantLR(BaseLR):
    def __init__(self, init_lr: float) -> None:
        super().__init__(init_lr=init_lr)

    def get_lr(self, step: int, optimizer: Optimizer) -> float:
        return self.init_lr


class WarmupLR(BaseLR):
    def __init__(self, init_lr: float, max_warmup_steps: int | None = None) -> None:
        super().__init__(init_lr=init_lr)
        if max_warmup_steps is None:
            warnings.warn("No warmup is happening when `max_warmup_steps` is not set.")
        else:
            assert max_warmup_steps > 0, "`max_warmup_steps` should be a positive integer"
        self.max_warmup_steps = max_warmup_steps

    def _should_warmup(self, step: int) -> bool:
        if self.max_warmup_steps and step < self.max_warmup_steps:
            return True
        return False

    def _warmup_step(self, step: int) -> float:
        if self.max_warmup_steps is not None:
            return self.init_lr * (step + 1) / self.max_warmup_steps
        return self.init_lr

    def get_lr(self, step: int, optimizer: torch.optim.Optimizer) -> float:
        if self._should_warmup(step):
            return self._warmup_step(step)
        return self.init_lr


class WarmupSchedulerLR(WarmupLR):
    def __init__(
        self,
        init_lr: float,
        min_lr: float = 0,
        max_decay_steps: int | None = None,
        max_warmup_steps: int | None = None,
        start_decay_at_step: int | None = None,
    ) -> None:
        super().__init__(init_lr=init_lr, max_warmup_steps=max_warmup_steps)
        assert min_lr >= 0, "`min_lr` is always positive."

        self.max_decay_steps = max_decay_steps
        self.start_decay_at_step = start_decay_at_step
        self.min_lr = min_lr

        self._strict_optim_lr_use: bool = False
        self._decay_counter: int = 0

    def _should_decay(self, step: int) -> bool:
        if self.max_decay_steps and self._decay_counter >= self.max_decay_steps:
            self._strict_optim_lr_use = False
            return False
        if self.start_decay_at_step and step < self.start_decay_at_step or step == 0:
            self._strict_optim_lr_use = True
            return False
        self._strict_optim_lr_use = False
        return True

    def _post_decay_return(self, optimizer: torch.optim.Optimizer) -> float:
        if self.min_lr > 0 and not self._strict_optim_lr_use:
            return self.min_lr
        return self._get_lr_from_optim(optimizer)


class ExponetialWarmupSchedulerLR(WarmupSchedulerLR):
    def __init__(
        self,
        init_lr: float,
        min_lr: float = 0,
        decay_rate: float | None = None,
        max_decay_steps: int | None = None,
        max_warmup_steps: int | None = None,
        start_decay_at_step: int | None = None,
    ) -> None:
        super().__init__(
            init_lr=init_lr,
            max_warmup_steps=max_warmup_steps,
            min_lr=min_lr,
            max_decay_steps=max_decay_steps,
            start_decay_at_step=start_decay_at_step,
        )
        # Dynamic decay_rate
        if decay_rate is None and max_warmup_steps and max_decay_steps:
            decay_steps = max_decay_steps
            log_rate = (math.log(min_lr) - math.log(init_lr)) / decay_steps
            decay_rate = pow(math.e, log_rate)
        assert decay_rate is not None, "decay_rate can't be None if max_warmup_steps and max_decay_steps are None"
        assert 0 < decay_rate < 1, "`decay_rate` should be between 0 and 1."
        self.decay_rate = decay_rate

    def get_lr(self, step: int, optimizer: torch.optim.Optimizer) -> float:
        if self._should_warmup(step):
            return self._warmup_step(step)
        if self._should_decay(step):
            self._decay_counter += 1
            decayed_lr = self._get_lr_from_optim(optimizer=optimizer) * self.decay_rate
            return max(self.min_lr, decayed_lr)
        return self._post_decay_return(optimizer)


class StepWarmupSchedulerLR(WarmupSchedulerLR):
    def __init__(
        self,
        init_lr: float,
        decay_rate: float,
        step_size: int,
        min_lr: float = 0,
        max_decay_steps: int | None = None,
        max_warmup_steps: int | None = None,
        start_decay_at_step: int | None = None,
    ) -> None:
        super().__init__(
            init_lr=init_lr,
            max_warmup_steps=max_warmup_steps,
            min_lr=min_lr,
            max_decay_steps=max_decay_steps,
            start_decay_at_step=start_decay_at_step,
        )
        assert step_size > 0, "step_size should be always greater than 0"

        self.decay_rate = decay_rate
        self.step_size = step_size

    def get_lr(self, step: int, optimizer: torch.optim.Optimizer) -> float:
        if self._should_warmup(step):
            return self._warmup_step(step)
        if self._should_decay(step):
            self._decay_counter += 1
            optim_lr = self._get_lr_from_optim(optimizer=optimizer)
            if (step) % self.step_size == 0:
                decayed_lr = optim_lr * self.decay_rate
                return max(self.min_lr, decayed_lr)
            return optim_lr
        return self._post_decay_return(optimizer)
