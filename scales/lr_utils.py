from __future__ import annotations

from typing import Any

import torch
from torch.optim.optimizer import Optimizer


# TODO: Implement LR Warmup
class LRDetails:
    def __init__(self, init_lr: float, lr_scheduler: str | None = None, **kwargs: Any) -> None:
        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler
        self.schduler_arg_dict = kwargs

    def instantiate_lr_scheduler(self, optimizer: Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if isinstance(self.lr_scheduler, str):
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler)
            return scheduler_cls(optimizer, **self.schduler_arg_dict)
        raise ValueError("The passed `lr_scheduler` is not of type string")
