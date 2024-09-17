from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from litgpt.config import Config
from litgpt.utils import num_parameters, save_config
from mup import get_shapes, make_base_shapes

from scales.lr_utils import LRScheduler
from scales.model import GPT_Scales


def save_checkpoint(
    fabric: L.Fabric,
    state: dict,
    checkpoint_dir: str | Path,
    train_step: int | None = None,
    recovery_state: bool = False,
    last_step: bool = False,
) -> None:
    checkpoint_name = "lit_model.pth"
    if train_step is not None:
        checkpoint_name = f"lit_model_{train_step}.pth"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving state to {str(checkpoint_dir)}")
    if fabric.global_rank == 0 and recovery_state and os.path.exists(Path(checkpoint_dir / checkpoint_name)):
        os.rename(Path(checkpoint_dir / checkpoint_name), Path(checkpoint_dir / "lit_model_recovery.pth"))
    if fabric.global_rank == 0 and last_step and recovery_state:
        os.remove(Path(checkpoint_dir / "lit_model_recovery.pth"))
    # This will save all torch related artifacts with their state_dict
    fabric.save(Path(checkpoint_dir / checkpoint_name), state)
    # TODO: save random state
    fabric.print(checkpoint_dir.parent)
    if fabric.global_rank == 0:
        save_config(state["model"].config, checkpoint_dir)


def load_checkpoint(
    fabric: L.Fabric,
    state: dict | None,
    checkpoint_dir: str | Path,
    recovery_state: bool = False,
    train_step: int | None = None,
) -> tuple[dict, Path]:
    checkpoint_name = "lit_model.pth"
    if train_step is not None:
        checkpoint_name = f"lit_model_{train_step}.pth"
    if recovery_state:
        checkpoint_name = "lit_model_recovery.pth"
    checkpoint_dir = Path(checkpoint_dir)
    fabric.print(f"Loading state from {str(checkpoint_dir / checkpoint_name)}")
    # This will load all torch related artifacts with their state dictionary,
    remainder = fabric.load(path=Path(checkpoint_dir / checkpoint_name), state=state)
    # TODO: Load random states
    model_config_path = Path(checkpoint_dir / "model_config.yaml")
    return remainder, model_config_path


def load_checkpoint_state(
    load_state_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    overwrite_checkpoint: bool = True,
) -> tuple[Any, nn.Module, torch.optim.Optimizer, Any]:
    # TODO: resolve checkpoint name when overwrite is False
    # if load_state_path is not a .pth file but a path, load the latest checkpoint (by steps)
    checkpoint_name = "checkpoint.pth"
    steps = None
    if load_state_path is not None:
        checkpoint = torch.load(load_state_path / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "train_steps" in checkpoint:
            steps = checkpoint["train_steps"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
    return steps, model, optimizer, scheduler


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


def total_gradient_l2_norm(model: nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def gradient_l2_norm_per_layer(model: nn.Module, global_step: int) -> dict:
    layer_grad_norms = defaultdict(list)
    for name, param in model.named_parameters():
        if "transformer.h" in name and param.grad is not None and param.requires_grad:
            layer_id = name.split(".transformer.h.")[-1].split(".")[0]  # extract the layer ID
            layer_grad_norms[layer_id].append(param.grad.detach().norm(2).item() ** 2)
    # calculating norm for each layer by summing the square of each parameter's gradient
    return {k: np.sum(v) ** 0.5 for k, v in layer_grad_norms.items()}


def weight_spectra(model: nn.Module) -> dict:
    singular_val_per_layer = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            singular_vals = torch.linalg.svdvals(mod.weight.data).detach()
            singular_val_per_layer[name] = singular_vals
    return singular_val_per_layer


def get_mup_shape_base(base_config: Config, target_config: Config, output_file: Path, verbose: bool = False) -> None:
    """Get the shape difference between two models with different scaling dimensions for muP.

    Refer to the `../examples/save_model_base_shape.py` script for more details.

    """
    base_model = get_shapes(GPT_Scales(base_config, mup_init=True))
    delta_model = get_shapes(GPT_Scales(target_config, mup_init=True))
    if isinstance(output_file, str):
        output_file = Path(output_file)
    make_base_shapes(base_model, delta_model, output_file)
    print(f"Scaling shape saved to {output_file.absolute()}!")
    if verbose:
        print(
            "\nNumber of base:target parameters (Kaplan): "
            f"{count_trainable_parameters_kaplan(GPT_Scales(base_config, mup_init=True)) / 1e6}M:"
            f"{count_trainable_parameters_kaplan(GPT_Scales(target_config, mup_init=True)) / 1e6}M"
        )
        print(
            "\nNumber of base:target parameters (Chinchilla): "
            f"{count_trainable_parameters_chinchilla(GPT_Scales(base_config, mup_init=True)) / 1e6}M:"
            f"{count_trainable_parameters_chinchilla(GPT_Scales(target_config, mup_init=True)) / 1e6}M"
        )
        print(
            "\nNumber of base:target parameters (LitGPT): "
            f"{num_parameters(GPT_Scales(base_config, mup_init=True), requires_grad=True) / 1e6}M:"
            f"{num_parameters(GPT_Scales(target_config, mup_init=True), requires_grad=True) / 1e6}M"
        )


def count_trainable_parameters_kaplan(model: GPT_Scales):
    """Count the number of parameters using the Kaplan approach.

    https://arxiv.org/abs/2001.08361

    Args:
    model : GPT model

    Returns:
    int: Total number of parameters

    """
    # TODO: verify code

    return 2 * model.config.n_embd * model.config.n_layer * (2 * model.config.n_embd + model.config.intermediate_size)


def count_trainable_parameters_chinchilla(
    model: nn.Module, return_all: bool = False, verbose: bool = False
) -> int | tuple[int, int]:
    """Count the number of parameters in a PyTorch model using an interpretation of the Chinchilla approach.

    Based on the Hoffmann et al. paper "Training Compute-Optimal Large Language Models",
    this function attempts to exclude embedding parameters. However, the exact definition
    of what constitutes "embedding parameters" may vary depending on the model architecture.

    NOTE: Generated from Claude 3.5 Sonnet on August 13, 2024.

    Args:
    model (nn.Module): PyTorch model

    Returns:
    int: Estimated number of non-embedding parameters
    or
    tuple[int, int]: Total number of parameters and embedding parameters

    """
    # TODO: verify code

    def is_embedding_like(module):
        return isinstance(module, (nn.Embedding, nn.EmbeddingBag))

    total_params = 0
    embedding_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # it's a leaf module
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if is_embedding_like(module):
                embedding_params += module_params
            else:
                total_params += module_params

    if verbose:
        print(f"Total parameters: {total_params + embedding_params}")
        print(f"Embedding parameters: {embedding_params}")
        print(f"Non-embedding parameters: {total_params}")

    if return_all:
        return total_params, embedding_params
    return embedding_params
