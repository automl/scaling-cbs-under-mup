from pathlib import Path
from typing import Tuple

import lightning as L
import torch.nn as nn
from litgpt.utils import save_config


def save_checkpoint(fabric: L.Fabric, state: dict, checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving state to {str(checkpoint_dir)}")
    fabric.save(Path(checkpoint_dir / "lit_model.pth"), state)
    print(checkpoint_dir.parent)
    if fabric.global_rank == 0:
        save_config(state["model"].config, checkpoint_dir)


def load_checkpoint(fabric: L.Fabric, checkpoint_dir: str | Path) -> Tuple[dict, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    fabric.print(f"Loading state from {str(checkpoint_dir)}")
    state = fabric.load(path=Path(checkpoint_dir / "lit_model.pth"))
    model_config_path = Path(checkpoint_dir / "model_config.yaml")
    return state, model_config_path


def total_gradient_norm(model: nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5
