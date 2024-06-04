"""This is a custom implementation for pretraining language models with litgpt and a simple version for getting
started."""

import time
import warnings
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from litgpt.model import GPT, Config
from litgpt.utils import CycleIterator, init_out_dir, num_parameters, parse_devices
from torch.utils.data import DataLoader

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
    hparams: dict = {"weight_decay": 0.02, "block_size": 2048},
    macro_batch_size: int = 4,
    micro_batch_size: int | None = None,
    nbr_steps_to_validate: int = 5,
    devices: int | str = "auto",
    max_norm: float | int | None = None,
    clip_val: float | int | None = None,
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
    fabric.launch()
    fabric.seed_everything(seed)

    if fabric.global_rank == 0:
        out_dir = init_out_dir(out_dir)

    if logging.log_dir is None:
        logging.log_dir = out_dir / "runs"

    device_count = parse_devices(devices=devices)

    fabric.print(f"Device count:{device_count}")
    fabric.print(f"Current strategy {fabric.strategy}")

    assert macro_batch_size % device_count == 0, "The macro batch size should be divisible by device count!"
    mini_batch_size = int(macro_batch_size / device_count)

    if micro_batch_size is not None:
        assert micro_batch_size > 0, "The `micro_batch_size` should be a positive integer!"
        assert mini_batch_size % micro_batch_size == 0, "mini batch size should be divisible by micro batch size!"
        accumulation_iters = int(mini_batch_size / micro_batch_size)
    else:
        micro_batch_size = mini_batch_size
        accumulation_iters = 1

    fabric.print(f"Accumulation iterations required:{accumulation_iters}")

    states = init_state(
        fabric=fabric,
        hparams=hparams,
        lr_details=lr_details,
        load_model_from_path=load_model_from_path,
        model_name=model_name,
        model_config_file=model_config_file,
    )

    model = states["model"]
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
    data.load_data_loaders(batch_size=micro_batch_size, block_size=block_size)

    train_dataloader = data.data_loaders["train"]
    val_dataloader = data.data_loaders["validation"]

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.print(f"Steps for training an epoch per device: {len(train_dataloader)}")
    fabric.print(f"Steps for validation per device: {len(val_dataloader)}")

    tokens_per_step = macro_batch_size * block_size
    max_data_tokens = len(train_dataloader) * tokens_per_step

    if force_unique_tokens:
        seen_tokens = states.get("train_tokens", 0)
        if (max_train_steps and max_train_steps * tokens_per_step > (max_data_tokens - seen_tokens)) or (
            max_train_tokens and max_train_tokens > (max_data_tokens - seen_tokens)
        ):
            raise ValueError(f"Training on Unique tokens can't be guaranteed available tokens: {max_data_tokens}")

    start_time = time.time()

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
        max_norm=max_norm,
        clip_val=clip_val,
        accumulation_iters=accumulation_iters,
    )
    end_time = time.time() - start_time
    print(f"Device {fabric.global_rank} - Train Time: {(end_time):.3f}s")
    avg_train_time = fabric.all_reduce(torch.tensor(end_time), reduce_op="mean")
    fabric.print(f"The average training time is {(avg_train_time):.3f}s")

    fabric.barrier()
    save_checkpoint(fabric=fabric, state=states, checkpoint_dir=out_dir)

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

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
    accumulation_iters: int = 1,
    max_norm: float | int | None = None,
    clip_val: float | int | None = None,
    max_train_steps: int | None = None,
    max_train_tokens: int | None = None,
) -> torch.Tensor:
    # TODO: Is `max_train_steps` a redundant arg if we have both `tokens_per_step` and `max_train_tokens`?
    if max_train_steps is None and max_train_tokens is None:
        raise ValueError("One of `max_train_steps` or `max_train_tokens` should be set.")
    if max_train_tokens and max_train_steps:
        raise ValueError("Only one of `max_train_steps` or `max_train_tokens` can be set.")

    train_iterator = CycleIterator(train_dataloader)
    loop_iters = 0
    device_running_loss = 0
    cumulative_time = 0

    # wait for all processes
    fabric.barrier()

    for batch in train_iterator:
        if max_train_steps is not None and states["train_steps"] >= max_train_steps:
            break
        if max_train_tokens is not None and states["train_tokens"] >= max_train_tokens:
            break

        if loop_iters == 0:
            start_time = time.time()

        is_accumulating = (loop_iters + 1) % accumulation_iters != 0

        for param_group in states["optimizer"].param_groups:
            param_group["lr"] = states["lr_details"].get_lr(states["train_steps"], optimizer=states["optimizer"])

        # Properly adjust the dimensions
        input_ids = batch[:, 0:max_seq_length].contiguous().long()
        targets = batch[:, 1 : (max_seq_length + 1)].contiguous().long()

        # synchronizes the gradients only after accumulation for speed
        with fabric.no_backward_sync(module=states["model"], enabled=is_accumulating):
            logits = states["model"](input_ids)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = nn.functional.cross_entropy(logits, targets)
            fabric.backward(loss / accumulation_iters)
            device_running_loss += loss.item() / accumulation_iters

        if accumulation_iters > 1 or fabric.world_size > 1:
            print(f"Loop Iteration {loop_iters} - Device {fabric.global_rank} - Micro Batch Loss {loss.item()} ")

        if not is_accumulating:
            if max_norm is not None or clip_val is not None:
                fabric.clip_gradients(states["model"], states["optimizer"], max_norm=max_norm, clip_val=clip_val)
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
            fabric.print(
                f"Train Step {states['train_steps']} - Tokens {states['train_tokens']} - Total Loss {loss.item()}"
                f" - Current Throughput {current_throughput} - Avg Throughput {avg_throughput}"
            )
            states["train_steps"] += 1
            device_running_loss = 0
            loop_iters = 0
        else:
            loop_iters += 1

        if (states["train_steps"] % nbr_steps_to_validate == 0 or states["train_steps"] == 1) and not is_accumulating:
            val_loss = validate(
                fabric,
                states["model"],
                val_dataloader,
                max_seq_length,
                max_val_steps,
            )
            fabric.print(f"Validation Loss: {val_loss}")

    final_val_loss = validate(
        fabric,
        states["model"],
        val_dataloader,
        max_seq_length,
        max_val_steps,
    )
    fabric.print(f"Final Validation Loss: {final_val_loss}")

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

    val_loss = torch.stack(val_losses).mean().detach().item()
    total_val_loss = fabric.all_reduce(torch.tensor(val_loss), reduce_op="mean")
    model.train()
    fabric.barrier()
    return total_val_loss
