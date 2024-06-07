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
from litgpt.utils import CycleIterator, init_out_dir
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scales.config.data_config import DataHandler
from scales.config.train_config import TrainConfig
from scales.utils import load_checkpoint, save_checkpoint, total_gradient_norm


def main(
    fabric: L.Fabric,
    data: DataHandler,
    train_args: TrainConfig,
    out_dir: Path = Path(__file__).parent.parent / "output",
) -> dict:
    fabric.seed_everything(train_args.seed)
    out_dir = init_out_dir(out_dir)
    logging = train_args.logging_args

    if logging.log_dir is None:
        logging.log_dir = out_dir / "runs"

    states = init_state(fabric=fabric, train_args=train_args)

    batch_size = train_args.batch_size
    block_size = train_args.block_size

    fabric.print(f"Number of trainable parameters: {train_args.trainable_params:,}")

    # Setting up the data with the relevant tokenizer
    data.load_data_loaders(batch_size=batch_size, block_size=block_size)

    train_dataloader = data.data_loaders["train"]
    val_dataloader = data.data_loaders["validation"]

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    fabric.print(f"Steps for training an epoch: {len(train_dataloader)}")
    fabric.print(f"Steps for validation: {len(val_dataloader)}")

    train_time = time.perf_counter()

    # Training and Validation
    val_loss = train(
        fabric=fabric,
        states=states,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_args=train_args,
    )
    fabric.print(f"Train time: {(time.perf_counter() - train_time):.3f}s")

    save_checkpoint(fabric, states, out_dir)

    return {"val_loss": val_loss}


def init_state(
    fabric: L.Fabric,
    train_args: TrainConfig,
    load_model_from_path: str | Path | None = None,
) -> dict:
    if load_model_from_path is None:
        states: Dict[str, Any] = {}
        model = GPT(train_args.model_config)
        lr_details = train_args.lr_scheduler
    else:
        if train_args.model_config_path or train_args.model_name:
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
        model.parameters(), lr=lr_details.init_lr, weight_decay=train_args.weight_decay, betas=(0.9, 0.95)
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
    train_args: TrainConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> torch.Tensor:
    max_seq_length = train_args.block_size
    logging = train_args.logging_args
    tokens_per_step = train_args.block_size * train_args.batch_size

    if logging.should_log():
        writer = SummaryWriter(log_dir=logging.log_dir)

    # Let's use cycleiterator to do max tokens...
    train_iterator = CycleIterator(train_dataloader)

    cumulative_time = 0.0

    for batch in train_iterator:
        if states["train_steps"] >= train_args.train_steps:
            break

        start_time = time.time()

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

        if train_args.max_norm is not None or train_args.clip_val is not None:
            fabric.clip_gradients(
                states["model"], states["optimizer"], max_norm=train_args.max_norm, clip_val=train_args.clip_val
            )

        if logging.total_gradient_norm and states["train_steps"] % logging.log_step == 0:
            total_norm = total_gradient_norm(states["model"])
            writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=states["train_steps"])

        states["optimizer"].step()

        batch_time = time.time() - start_time
        cumulative_time += batch_time

        # Calculate running average throughput
        avg_batch_time = cumulative_time / (states["train_steps"] + 1)
        avg_throughput = tokens_per_step / avg_batch_time

        states["train_tokens"] += tokens_per_step
        fabric.print(
            f"Train Step {states['train_steps']} - Tokens {states['train_tokens']} - Loss {loss}"
            f" - Batch Time: {batch_time} - Average Throughput: {avg_throughput}"
        )

        if logging.train_loss and states["train_steps"] % logging.log_step == 0:
            writer.add_scalar(tag="Train Loss", scalar_value=loss, global_step=states["train_steps"])

        if states["train_steps"] % train_args.validate_every == 0 or states["train_steps"] == 0:
            val_loss = validate(
                fabric,
                states["model"],
                val_dataloader,
                max_seq_length,
                train_args.max_val_steps,
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
        total_norm = total_gradient_norm(states["model"])
        writer.add_scalar(tag="Total Gradient Norm", scalar_value=total_norm, global_step=states["train_steps"])

    final_val_loss = validate(
        fabric,
        states["model"],
        val_dataloader,
        max_seq_length,
        train_args.max_val_steps,
    )
    fabric.print(f"Final Validation Loss: {final_val_loss}")

    if logging.validation_loss:
        writer.add_scalar(tag="Validation Loss", scalar_value=val_loss, global_step=states["train_steps"])

    if logging.should_log():
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
