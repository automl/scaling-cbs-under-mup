"""This is a custom implementation for pretraining language models with litgpt and a simple version for getting
started."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict
from warnings import warn

import lightning as L
import torch
import torch.nn as nn
import yaml
from litgpt.utils import CycleIterator, init_out_dir
from mup import MuAdamW, set_base_shapes
from torch.utils.data import DataLoader

from scales.config.data_config import DataHandler
from scales.config.train_config import TrainConfig
from scales.model import GPT_Scales, file_data_share, initialize_weights
from scales.tblog_utils import load_tb
from scales.utils import load_checkpoint, save_checkpoint


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
    states = init_state(
        fabric=fabric,
        train_args=train_args,
        out_dir=out_dir,
    )

    device_count = train_args.devices

    fabric.print(f"Device count:{device_count}")
    fabric.print(f"Current strategy {fabric.strategy}")

    assert train_args.accumulation_iters > 0, "`accumulation_iters` should be a positive integer"

    effective_batch_size = int(device_count * train_args.accumulation_iters * train_args.micro_batch_size)
    fabric.print(f"Effective batch size for this setup is {effective_batch_size}")

    micro_batch_size = train_args.micro_batch_size
    block_size = train_args.block_size

    fabric.print(f"Number of trainable parameters: {train_args.trainable_params:,}")

    # Setting up the data with the relevant tokenizer
    data.load_data_loaders(
        batch_size=micro_batch_size,
        block_size=block_size,
        access_internet=access_internet,
    )

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
            "d_model": train_args.model_config.d_model,  # type: ignore
            "n_head": train_args.model_config.n_head,  # type: ignore
            "n_layer": train_args.model_config.n_layer,  # type: ignore
            "block_size": train_args.block_size,
        },
    }
    with open(out_dir / "info.yaml", "w") as f:
        yaml.dump(_info, f)

    # Note, this train_args is NOT the same as the one passed to the function
    train_args.write_yaml(out_dir / "train_config_post_init.yaml", ignore_defaults=False)

    train_time = time.time()

    # Training and Validation
    result = train(
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
    df = load_tb(output_dir=out_dir, train_config_file_name="train_config_post_init")
    df.to_csv(str(out_dir / "tb_logs.csv"))

    if train_args.save_state_path is not None:
        result_path = train_args.save_state_path / "result.yaml"
        with result_path.open(mode="w", encoding="utf-8") as file:
            yaml.dump(result, file)

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    return result


def init_state(
    fabric: L.Fabric,
    train_args: TrainConfig,
    out_dir: Path,
) -> dict:
    """Initialize the state for training.

    Args:
        fabric: The fabric object
        train_args: The training configuration
        out_dir: The output directory where the logs and checkpoints will be saved
            If `save_state_path` is not provided, the state checkpoints will be saved here

    Returns:
        dict: The state for training

    """
    if train_args.logging_args.log_dir is None:
        train_args.logging_args.log_dir = out_dir / "logs"

    train_args.logging_args.start_logger()
    # accessing this attribute here so base and delta models
    # can be garbage collected before the target model is loaded
    mup_base_shape = train_args.mup_base_shape
    states: Dict[str, Any] = {"train_tokens": 0, "train_steps": 0, "torch_scheduler": train_args.lr_scheduler.scheduler}
    states["model"] = GPT_Scales(train_args.model_config, mup_init=mup_base_shape is not None)

    if mup_base_shape is not None:
        set_base_shapes(states["model"], mup_base_shape, rescale_params=train_args.load_state_path is None)

    if train_args.deepseek_hparams:
        fabric.print(f"Changing weight_init_type from {train_args.weight_init_type} to DeepSeek")
        train_args.weight_init_type = "DeepSeek"

    # Note: Does not work when setting a path for base shape mup. Maybe remove the argument and use the other arg
    initialize_weights(
        fabric=fabric,
        model=states["model"],
        mup_base_scales=train_args.mup_base_scales,
        init_type=train_args.weight_init_type,
    )

    if train_args.lr_scheduler is None:
        raise ValueError("Please provide an appropriate learning rate configuration.")

    if train_args.mup_base_shape:
        fabric.print("Using MuP Optimizer")
        states["optimizer"] = MuAdamW(
            states["model"].parameters(),
            lr=train_args.lr_scheduler.init_lr,
            weight_decay=train_args.true_weight_decay,
            betas=(train_args.adam_beta_1, train_args.adam_beta_2),
            eps=train_args.adam_eps,
        )
    else:
        states["optimizer"] = torch.optim.AdamW(
            states["model"].parameters(),
            lr=train_args.lr_scheduler.init_lr,
            weight_decay=train_args.true_weight_decay,
            betas=(train_args.adam_beta_1, train_args.adam_beta_2),
            eps=train_args.adam_eps,
        )

    states["model"] = fabric.setup_module(states["model"])
    states["optimizer"] = fabric.setup_optimizers(states["optimizer"])

    if train_args.save_state_path is None:
        train_args.save_state_path = out_dir

    if train_args.save_init_state:
        save_checkpoint(
            fabric, state=states, train_step=states["train_steps"], checkpoint_dir=Path(train_args.save_state_path)
        )

    # load checkpoint state
    if train_args.load_state_path is not None:
        # load the state here
        try:
            _, _ = load_checkpoint(fabric, states, Path(train_args.load_state_path))
        except FileNotFoundError:
            warn(f"The path {train_args.load_state_path} does not exist, no checkpoint running")
        except Exception as e:
            warn(f"An error occurred while loading the checkpoint: {e}")

    return states


def train(
    fabric: L.Fabric,
    states: dict,
    train_args: TrainConfig,
    effective_batch_size: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> dict[str, int | list[float] | None]:
    max_seq_length = train_args.block_size
    logger = train_args.logging_args
    tokens_per_step = train_args.block_size * effective_batch_size
    accumulation_iters = train_args.accumulation_iters

    result: dict[str, int | list[float] | None] = {"train_steps": 0, "val_loss": None, "train_loss": None}
    # Let's use cycleiterator to do max tokens...
    train_iterator = CycleIterator(train_dataloader)

    loop_iters = 0
    device_running_loss = 0
    cumulative_time = 0
    last_step = False
    exit_loop = False

    # wait for all processes
    fabric.barrier()

    # accounting for steps gone in the previous training
    for i, _batch in enumerate(train_iterator):
        if i == states["train_steps"]:
            break

    # main training loop
    for batch in train_iterator:
        is_accumulating = (loop_iters + 1) % accumulation_iters != 0

        if states["train_steps"] + 1 >= train_args.train_steps:
            last_step = True

        if loop_iters == 0:
            start_time = time.time()

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
                last=last_step,
            )
            logger.max_attention_logits_per_layer(
                attn_logits=file_data_share.layer_wise_max_attn_weight,
                step=states["train_steps"],
                fabric=fabric,
                is_accumulating=is_accumulating,
                last=last_step,
            )
            logger.max_attention_logits_all(
                attn_logits=file_data_share.layer_wise_max_attn_weight,
                step=states["train_steps"],
                fabric=fabric,
                is_accumulating=is_accumulating,
                last=last_step,
            )
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            loss = nn.functional.cross_entropy(logits, targets)

            if getattr(train_args, "z_loss_eps", None) is not None:
                # implementation from
                # https://github.com/mlfoundations/open_lm/blob/c0f131958abeab17b691930c5182cc9abe74e37b/open_lm/losses.py#L21
                z_loss = train_args.z_loss_eps * torch.square(torch.logsumexp(logits, dim=-1)).mean()
                loss += z_loss
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
            logger.learning_rate(optimizer=states["optimizer"], step=states["train_steps"], last=last_step)
            logger.weight_spectra_max(model=states["model"], step=states["train_steps"], last=last_step)
            logger.weight_spectra_diff(model=states["model"], step=states["train_steps"], last=last_step)

            states["optimizer"].step()
            logger.optimizer_stats(step=states["train_steps"], optimizer=states["optimizer"])
            states["optimizer"].zero_grad()
            states["train_tokens"] += tokens_per_step

            train_args.lr_scheduler.step(steps=states["train_steps"] + 1, optimizer=states["optimizer"])
            end_time = time.time()

            total_batch_time = fabric.all_reduce(torch.tensor(end_time - start_time), reduce_op="mean")
            current_throughput = tokens_per_step / total_batch_time

            cumulative_time += total_batch_time
            avg_batch_time = cumulative_time / (states["train_steps"] + 1)
            avg_throughput = tokens_per_step / avg_batch_time
            # get the mean loss from all devices
            loss = fabric.all_reduce(torch.tensor(device_running_loss), reduce_op="mean")
            result["train_loss"] = loss.item()
            logger.train_loss(loss=loss, step=states["train_steps"], last=last_step)
            fabric.print(
                f"Train Step {states['train_steps']} - Tokens {states['train_tokens']} - Total Loss {loss.item()}"
                f" - Current Throughput {current_throughput} - Avg Throughput {avg_throughput}"
            )
            states["train_steps"] += 1
            device_running_loss = 0
            loop_iters = 0
            if last_step is True:
                exit_loop = True
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
            result["train_steps"] = states["train_steps"]
            result["val_loss"] = val_loss.detach().cpu().numpy().tolist()

        # checkpoint saving
        if (
            states["train_steps"] % train_args.save_state_every == 0 or states["train_steps"] == 1 or last_step is True
        ) and not is_accumulating:
            states["torch_scheduler"] = train_args.lr_scheduler.scheduler
            save_checkpoint(fabric, state=states, checkpoint_dir=Path(str(train_args.save_state_path)))
            # save_checkpoint_state(
            #     save_state_path=Path(str(train_args.save_state_path)),
            #     train_steps=states["train_steps"],
            #     model=states["model"],
            #     optimizer=states["optimizer"],
            #     scheduler=states["torch_scheduler"],
            #     overwrite_checkpoint=train_args.overwrite_state,
            # )
            if train_args.save_state_path is not None:
                result_path = train_args.save_state_path / "result.yaml"
                with result_path.open(mode="w", encoding="utf-8") as file:
                    yaml.dump(result, file)

        file_data_share.clear_data()

        if exit_loop is True:
            break
    # end of training loop
    logger.close()

    return result


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
