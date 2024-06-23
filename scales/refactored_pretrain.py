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
import yaml
from litgpt.model import GPT, Config
from litgpt.utils import CycleIterator, init_out_dir, parse_devices
from torch.utils.data import DataLoader

# from scales.lr_utils import LRScheduler
from scales.args import LoggingArgs
from scales.config.data_config import DataHandler
from scales.config.train_config import TrainConfig
from scales.utils import load_checkpoint, load_checkpoint_state, save_checkpoint, save_checkpoint_state


def main(
    fabric: L.Fabric,
    data: DataHandler,
    train_args: TrainConfig,
    out_dir: Path = Path(__file__).parent.parent / "output",
    access_internet: bool = True,
) -> dict:
    fabric.launch()
    fabric.seed_everything(train_args.seed)
    out_dir = init_out_dir(out_dir)

    # Initialize state
    states = init_state(fabric=fabric, train_args=train_args, out_dir=out_dir)

    logger = train_args.logging_args

    if logger.log_dir is None:
        logger.update_logdir(out_dir / "logs")

    device_count = parse_devices(devices=train_args.devices)

    fabric.print(f"Device count:{device_count}")
    fabric.print(f"Current strategy {fabric.strategy}")

    assert train_args.accumulation_iters > 0, "`accumulation_iters` should be a positive integer"

    effective_batch_size = device_count * train_args.accumulation_iters * train_args.micro_batch_size
    fabric.print(f"Effective batch size for this setup is {effective_batch_size}")

    micro_batch_size = train_args.micro_batch_size
    block_size = train_args.block_size

    fabric.print(f"Number of trainable parameters: {train_args.trainable_params:,}")

    # Setting up the data with the relevant tokenizer
    data.load_data_loaders(batch_size=micro_batch_size, block_size=block_size, access_internet=access_internet)

    train_dataloader = data.data_loaders["train"]
    val_dataloader = data.data_loaders["validation"]

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    fabric.print(f"Steps for training an epoch per device: {len(train_dataloader)}")
    fabric.print(f"Steps for validation per device: {len(val_dataloader)}")

    _info = {
        "parameters": train_args.trainable_params,
        "effective_batch_size": effective_batch_size,
        "devices": device_count,
        "scales": {
            "d_model": train_args.model_config.d_model,
            "n_head": train_args.model_config.n_head,
            "n_layer": train_args.model_config.n_layer,
            "block_size": train_args.block_size,
        }
    }
    with open(out_dir / "info.yaml", "w") as f:
        yaml.dump(_info, f)

    train_time = time.time()

    # Training and Validation
    val_loss = train(
        fabric=fabric,
        states=states,
        effective_batch_size=effective_batch_size,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_args=train_args,
    )
    end_time = time.time() - train_time
    print(f"Device {fabric.global_rank} - Train Time: {(end_time):.3f}s")
    avg_train_time = fabric.all_reduce(torch.tensor(end_time), reduce_op="mean")
    fabric.print(f"The average training time is {(avg_train_time):.3f}s")

    fabric.barrier()
    save_checkpoint(fabric=fabric, state=states, checkpoint_dir=out_dir)

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    return {"val_loss": val_loss.numpy().tolist()}


def init_state(
    fabric: L.Fabric,
    train_args: TrainConfig,
    out_dir: Path,
    save_init_state: bool = True,
    load_model_from_path: str | Path | None = None,
) -> dict:
    """Initialize the state for training.

    Args:
        fabric: The fabric object
        train_args: The training configuration
        out_dir: The output directory where the logs and checkpoints will be saved
            If `save_state_path` is not provided, the state checkpoints will be saved here
        save_init_state: Whether to save the initial state, especially the initialization weights
        load_model_from_path: The path to load the model from (TODO: write what is different here)

    Returns:
        dict: The state for training

    """
    train_args.logging_args = LoggingArgs(
        tracked_metrics=train_args.tracked_metrics,
        global_log_step=train_args.global_log_step,
        log_dir=train_args.log_dir,
    )

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
        train_args.lr_scheduler = states["lr_scheduler"]
        model.load_state_dict(states["model"])

    if lr_details is None:
        raise ValueError("Please provide an appropriate learning rate configuration.")

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_details.init_lr, weight_decay=train_args.weight_decay, betas=(0.9, 0.95)
    )

    if train_args.lr_scheduler.torch_scheduler is not None:
        torch_scheduler = train_args.lr_scheduler._instantiate_lr_scheduler(optimizer=optimizer)
    else:
        torch_scheduler = None
    if len(states) != 0 and torch_scheduler is not None:
        torch_scheduler.load_state_dict(states["torch_scheduler"])

    if len(states) != 0:
        optimizer.load_state_dict(states["optimizer"])
    optimizer = fabric.setup_optimizers(optimizer)

    if len(states) == 0:
        states["train_steps"] = 0
        states["train_tokens"] = 0

    if train_args.save_state_path is None:
        train_args.save_state_path = out_dir

    if save_init_state:
        save_checkpoint_state(
            save_state_path=Path(train_args.save_state_path),
            train_steps=states["train_steps"],
            model=model,
            optimizer=optimizer,
            scheduler=torch_scheduler,
            overwrite_checkpoint=False,  # adds a step to the checkpoint name, 0 in this case
        )

    # load checkpoint state
    train_steps = 0
    if train_args.load_state_path is not None:
        # load the state here
        train_steps, model, optimizer = load_checkpoint_state(
            load_state_path=Path(train_args.load_state_path),
            model=model,
            optimizer=optimizer,
            scheduler=torch_scheduler,
            overwrite_checkpoint=train_args.overwrite_state,
        )

    states["train_steps"] = train_steps
    states["model"] = model
    states["optimizer"] = optimizer
    states["lr_scheduler"] = train_args.lr_scheduler
    states["torch_scheduler"] = torch_scheduler

    return states


def train(
    fabric: L.Fabric,
    states: dict,
    train_args: TrainConfig,
    effective_batch_size: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> torch.Tensor:
    max_seq_length = train_args.block_size
    logger = train_args.logging_args
    tokens_per_step = train_args.block_size * effective_batch_size
    accumulation_iters = train_args.accumulation_iters

    # Let's use cycleiterator to do max tokens...
    train_iterator = CycleIterator(train_dataloader)

    loop_iters = 0
    device_running_loss = 0
    cumulative_time = 0
    last_step = False

    # wait for all processes
    fabric.barrier()

    # accounting for steps gone in the previous training
    for i, _batch in enumerate(train_iterator):
        if i == states["train_steps"]:
            break

    # main training loop
    for batch in train_iterator:
        if states["train_steps"] + 1 == train_args.train_steps:
            last_step = True
        elif states["train_steps"] > train_args.train_steps:
            raise ValueError("Something unexpected during training has led to more train steps.")

        if loop_iters == 0:
            start_time = time.time()

        is_accumulating = (loop_iters + 1) % accumulation_iters != 0

        # Properly adjust the dimensions
        input_ids = batch[:, 0:max_seq_length].contiguous().long()
        targets = batch[:, 1 : (max_seq_length + 1)].contiguous().long()

        with fabric.no_backward_sync(module=states["model"], enabled=is_accumulating):
            logits = states["model"](input_ids)
            logger.output_logits_mean(
                logits=logits,
                step=states["train_steps"],
                fabric=fabric,
                is_accumulating=is_accumulating,
                accumulation_iters=accumulation_iters,
                last=last_step,
            )
            logger.output_logits_max(
                logits=logits,
                step=states["train_steps"],
                fabric=fabric,
                is_accumulating=is_accumulating,
                accumulation_iters=accumulation_iters,
                last=last_step,
            )
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = nn.functional.cross_entropy(logits, targets)
            fabric.backward(loss / accumulation_iters)
            device_running_loss += loss.item() / accumulation_iters

        if accumulation_iters > 1 or fabric.world_size > 1:
            print(f"Loop Iteration {loop_iters} - Device {fabric.global_rank} - Micro Batch Loss {loss.item()} ")

        if not is_accumulating:
            if train_args.clip_max_norm is not None or train_args.clip_max_val is not None:
                fabric.clip_gradients(
                    states["model"],
                    states["optimizer"],
                    max_norm=train_args.clip_max_norm,
                    clip_val=train_args.clip_max_val,
                )
            logger.total_gradient_norm(model=states["model"], step=states["train_steps"], last=last_step)
            logger.gradient_norm_per_layer(model=states["model"], step=states["train_steps"], last=last_step)
            states["lr_scheduler"].step(
                steps=states["train_steps"], optimizer=states["optimizer"], scheduler=states["torch_scheduler"]
            )
            logger.learning_rate(optimizer=states["optimizer"], step=states["train_steps"], last=last_step)
            states["optimizer"].step()
            states["optimizer"].zero_grad()
            states["train_tokens"] += tokens_per_step
            end_time = time.time()

            total_batch_time = fabric.all_reduce(torch.tensor(end_time - start_time), reduce_op="mean")
            current_throughput = tokens_per_step / total_batch_time

            cumulative_time += total_batch_time
            avg_batch_time = cumulative_time / (states["train_steps"] + 1)
            avg_throughput = tokens_per_step / avg_batch_time
            # get the mean loss from all devices
            loss = fabric.all_reduce(torch.tensor(device_running_loss), reduce_op="mean")
            logger.train_loss(loss=loss, step=states["train_steps"], last=last_step)
            fabric.print(
                f"Train Step {states['train_steps']} - Tokens {states['train_tokens']} - Total Loss {loss.item()}"
                f" - Current Throughput {current_throughput} - Avg Throughput {avg_throughput}"
            )
            states["train_steps"] += 1
            device_running_loss = 0
            loop_iters = 0
        else:
            loop_iters += 1

        # validation loop
        if (
            states["train_steps"] % train_args.validate_every == 0 or states["train_steps"] == 1 or last_step is True
        ) and not is_accumulating:
            val_loss = validate(
                fabric,
                states["model"],
                val_dataloader,
                max_seq_length,
                train_args.max_val_steps,
            )
            logger.validation_loss(val_loss, step=states["train_steps"], last=last_step)
            fabric.print(f"Validation Loss: {val_loss}")

        # checkpoint saving
        if (
            states["train_steps"] % train_args.save_state_every == 0 or states["train_steps"] == 1 or last_step is True
        ) and not is_accumulating:
            save_checkpoint_state(
                save_state_path=Path(train_args.save_state_path),
                train_steps=states["train_steps"],
                model=states["model"],
                optimizer=states["optimizer"],
                scheduler=states["torch_scheduler"],
                overwrite_checkpoint=train_args.overwrite_state,
            )

        if last_step is True:
            break
    # end of training loop
    logger.close()

    return val_loss.detach().cpu()


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

    val_loss = torch.stack(val_losses).mean().detach().item()
    total_val_loss = fabric.all_reduce(torch.tensor(val_loss), reduce_op="mean")
    model.train()
    fabric.barrier()
    return total_val_loss
