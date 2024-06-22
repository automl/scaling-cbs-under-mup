from pathlib import Path
import numpy as np
import random
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from litgpt.utils import save_config

from scales.lr_utils import LRScheduler


def save_checkpoint(fabric: L.Fabric, state: dict, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving state to {str(checkpoint_dir)}")
    fabric.save(Path(checkpoint_dir / "lit_model.pth"), state)
    fabric.print(checkpoint_dir.parent)
    if fabric.global_rank == 0:
        save_config(state["model"].config, checkpoint_dir)


def load_checkpoint(fabric: L.Fabric, checkpoint_dir: str | Path) -> Tuple[dict, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    fabric.print(f"Loading state from {str(checkpoint_dir)}")
    state = fabric.load(path=Path(checkpoint_dir / "lit_model.pth"))
    model_config_path = Path(checkpoint_dir / "model_config.yaml")
    return state, model_config_path


def load_checkpoint_state(
    load_state_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    overwrite_checkpoint: bool = True,
) -> Tuple[int, nn.Module, torch.optim.Optimizer]:
    # TODO: resolve checkpoint name when overwrite is False
    # if load_state_path is not a .pth file but a path, load the latest checkpoint (by steps)
    checkpoint_name = "checkpoint.pth"
    steps = None
    if load_state_path is not None:
        checkpoint = torch.load(load_state_path / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if "train_steps" in checkpoint:
            steps = checkpoint["train_steps"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
    return steps, model, optimizer


def save_checkpoint_state(
    save_state_path: Path,
    train_steps: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    overwrite_checkpoint: bool = True,
) -> None:
    checkpoint_name = "checkpoint.pth" if overwrite_checkpoint else f"checkpoint_{train_steps}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "train_steps": train_steps,
        },
        save_state_path / checkpoint_name,
    )



def total_gradient_norm(model: nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5

