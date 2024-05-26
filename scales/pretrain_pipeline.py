"""This is a custom implementation for pretraining language models with litgpt and a simple version for getting
started.

Current missing features:

- No precision editing from fabric
- No logger (only to terminal)
- No grad accumulation

"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from litgpt.model import GPT, Config
from litgpt.utils import CycleIterator, init_out_dir, num_parameters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scales.args import LoggingArgs
from scales.data_utils import DataHandler
from scales.lr_utils import BaseLR
from scales.utils import load_checkpoint, save_checkpoint


def main(
    fabric: L.Fabric,
    data: DataHandler,
    logging: LoggingArgs = LoggingArgs(),
    lr_details: BaseLR | None = None,
    out_dir: Path = Path(__file__).parent.parent / "output",
    hparams: dict = {"weight_decay": 0.02, "batch_size": 64, "block_size": 2048},
    nbr_steps_to_validate: int = 5,
    load_model_from_path: str | Path | None = None,
    max_train_steps: int | None = None,
    max_val_steps: int = 5,
    max_train_tokens: int | None = None,
    tokens_per_param: int | None = None,
    force_unique_tokens: bool = False,
    model_name: str | None = None,
    model_config_file: str | Path | None = None,
    seed: int = 1337,
) -> dict:
    fabric.seed_everything(seed)
    out_dir = init_out_dir(out_dir)

    if logging.log_dir is None:
        logging.log_dir = out_dir / "runs"

    states = init_state(
        fabric=fabric,
        hparams=hparams,
        lr_details=lr_details,
        load_model_from_path=load_model_from_path,
        model_name=model_name,
        model_config_file=model_config_file,
    )

    model = states["model"]
    batch_size = hparams.get("batch_size", 64)
    block_size = hparams.get("block_size", 2048)

    trainable_params = num_parameters(model, requires_grad=True)
    if tokens_per_param is None and max_train_tokens is None and max_train_steps is None:
        raise ValueError("One of `max_train_steps` or `max_train_tokens` or `tokens_per_param` should be set.")
    if tokens_per_param and (max_train_tokens or max_train_steps):
        raise ValueError("`tokens_per_param` can be set together with `max_train_tokens` or `max_train_steps`")
    if tokens_per_param:
        max_train_tokens = tokens_per_param * trainable_params

    fabric.print(f"Number of trainable parameters: {trainable_params:,}")

    # Setting up the data with the relevant tokenizer
    data.load_data_loaders(batch_size=batch_size, block_size=block_size)

    train_dataloader = data.data_loaders["train"]
    val_dataloader = data.data_loaders["validation"]

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    fabric.print(f"Steps for training an epoch: {len(train_dataloader)}")
    fabric.print(f"Steps for validation: {len(val_dataloader)}")

    tokens_per_step = batch_size * block_size
    max_data_tokens = len(train_dataloader) * tokens_per_step

    if force_unique_tokens:
        seen_tokens = states.get("train_tokens", 0)
        if (max_train_steps and max_train_steps * tokens_per_step > (max_data_tokens - seen_tokens)) or (
            max_train_tokens and max_train_tokens > (max_data_tokens - seen_tokens)
        ):
            raise ValueError(f"Training on Unique tokens can't be guaranteed available tokens: {max_data_tokens}")

    train_time = time.perf_counter()

    # Training and Validation
    val_loss = train(
        fabric=fabric,
        states=states,
        tokens_per_step=tokens_per_step,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_val_steps=max_val_steps,
        max_seq_length=block_size,
        max_train_steps=max_train_steps,
        max_train_tokens=max_train_tokens,
        nbr_steps_to_validate=nbr_steps_to_validate,
        logging=logging,
    )
    fabric.print(f"Train time: {(time.perf_counter() - train_time):.3f}s")

    save_checkpoint(fabric, states, out_dir)

    return {"val_loss": val_loss}


def init_state(
    fabric: L.Fabric,
    lr_details: BaseLR | None = None,
    hparams: dict | None = None,
    load_model_from_path: str | Path | None = None,
    model_name: str | None = None,
    model_config_file: str | Path | None = None,
) -> dict:
    if hparams is None:
        hparams = {"lr": 3e-3, "weight_decay": 0.02}

    if load_model_from_path is None:
        # Setting up model configuration
        if model_config_file and model_name is None:
            config = Config.from_file(model_config_file)
        elif model_name and model_config_file is None:
            config = Config.from_name(model_name)
        elif model_config_file and model_name:
            raise ValueError("Only one of `model_name` or `model_config` can be set.")
        else:
            raise ValueError("Please specify `model_name` or `model_config_file`")
        states: Dict[str, Any] = {}
        model = GPT(config)
    else:
        if model_config_file or model_name:
            warnings.warn(
                "The configuration yaml in the loaded directory will be used "
                "`model_config_file` and `model_name` are ignored"
            )
        states, model_path = load_checkpoint(fabric, load_model_from_path)
        config = Config.from_file(model_path)
        model = GPT(config)
        lr_details = states["lr_details"]
        model.load_state_dict(states["model"])

    if lr_details is None:
        raise ValueError("Please provide an appropriate learning rate configuration.")

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_details.init_lr, weight_decay=hparams["weight_decay"], betas=(0.9, 0.95)
    )
    if len(states) != 0:
        optimizer.load_state_dict(states["optimizer"])
    optimizer = fabric.setup_optimizers(optimizer)

    if len(states) == 0:
        states["train_steps"] = 0
        states["train_tokens"] = 0

    states["model"] = model
    states["optimizer"] = optimizer
    states["lr_details"] = lr_details

    return states


def train(
    fabric: L.Fabric,
    states: dict,
    tokens_per_step: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_val_steps: int,
    max_seq_length: int,
    logging: LoggingArgs,
    nbr_steps_to_validate: int = 5,
    max_train_steps: int | None = None,
    max_train_tokens: int | None = None,
) -> torch.Tensor:
    # TODO: Is `max_train_steps` a redundant arg if we have both `tokens_per_step` and `max_train_tokens`?
    if max_train_steps is None and max_train_tokens is None:
        raise ValueError("One of `max_train_steps` or `max_train_tokens` should be set.")
    if max_train_tokens and max_train_steps:
        raise ValueError("Only one of `max_train_steps` or `max_train_tokens` can be set.")

    if logging.should_log():
        writer = SummaryWriter(log_dir=logging.log_dir)

    # Let's use cycleiterator to do max tokens...
    train_iterator = CycleIterator(train_dataloader)

    for batch in train_iterator:
        if max_train_steps is not None and states["train_steps"] >= max_train_steps:
            break
        if max_train_tokens is not None and states["train_tokens"] >= max_train_tokens:
            break

        states["optimizer"].zero_grad()

        for param_group in states["optimizer"].param_groups:
            param_group["lr"] = states["lr_details"].get_lr(states["train_steps"], optimizer=states["optimizer"])
            if logging.learning_rate and states["train_steps"] % logging.log_step == 0:
                writer.add_scalar(
                    tag="Learning Rate", scalar_value=param_group["lr"], global_step=states["train_steps"]
                )

        # Properly adjust the dimensions
        input_ids = batch[:, 0:max_seq_length].contiguous().long()
        targets = batch[:, 1 : (max_seq_length + 1)].contiguous().long()
        logits = states["model"](input_ids)

        if logging.output_logits_mean and states["train_steps"] % logging.log_step == 0:
            logits_mean = logits.mean().item()
            writer.add_scalar(tag="Output Logits Mean", scalar_value=logits_mean, global_step=states["train_steps"])

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = nn.functional.cross_entropy(logits, targets)

        # update weights
        fabric.backward(loss)

        if logging.total_gradient_norm and states["train_steps"] % logging.log_step == 0:
            total_norm = 0
            parameters = [p for p in states["model"].parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=states["train_steps"])

        states["optimizer"].step()

        states["train_tokens"] += tokens_per_step
        fabric.print(f"Train Step {states['train_steps']} - Tokens {states['train_tokens']} - Loss {loss}")

        if logging.train_loss and states["train_steps"] % logging.log_step == 0:
            writer.add_scalar(tag="Train Loss", scalar_value=loss, global_step=states["train_steps"])

        if states["train_steps"] % nbr_steps_to_validate == 0 or states["train_steps"] == 0:
            val_loss = validate(
                fabric,
                states["model"],
                val_dataloader,
                max_seq_length,
                max_val_steps,
            )
            fabric.print(f"Validation Loss: {val_loss}")

            if logging.validation_loss and states["train_steps"] % logging.log_step == 0:
                writer.add_scalar(tag="Validation Loss", scalar_value=val_loss, global_step=states["train_steps"])

        states["train_steps"] += 1

    if logging.train_loss:
        writer.add_scalar(tag="Train Loss", scalar_value=loss, global_step=states["train_steps"])

    if logging.output_logits_mean:
        logits_mean = logits.mean().item()
        writer.add_scalar(tag="Output Logits Mean", scalar_value=logits_mean, global_step=states["train_steps"])

    if logging.total_gradient_norm:
        total_norm = 0
        parameters = [p for p in states["model"].parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=states["train_steps"])

    final_val_loss = validate(
        fabric,
        states["model"],
        val_dataloader,
        max_seq_length,
        max_val_steps,
    )
    fabric.print(f"Final Validation Loss: {final_val_loss}")

    if logging.validation_loss:
        writer.add_scalar(tag="Validation Loss", scalar_value=val_loss, global_step=states["train_steps"])

    writer.close()

    return final_val_loss


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    max_seq_length: int,
    max_val_steps: int | None = None,
) -> torch.Tensor:
    fabric.barrier()
    model.eval()

    step = 0
    val_losses = []
    for step, batch in enumerate(val_dataloader):
        if max_val_steps and step >= max_val_steps:
            break
        input_ids = batch[:, :max_seq_length].contiguous().long()
        targets = batch[:, 1 : max_seq_length + 1].contiguous().long()
        logits = model(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = nn.functional.cross_entropy(logits, targets)
        val_losses.append(loss)
    else:
        # if no break is taken
        fabric.print(f"Validation data is exhausted in {step} steps")

    val_loss = torch.stack(val_losses).mean()
    model.train()
    fabric.barrier()
    return val_loss
