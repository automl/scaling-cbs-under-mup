"""
This is a custom implementation for pretraining language models with litgpt and
a simple version for getting started. 

Current missing features:

- No saving and restarting training
- No precision editing from fabric
- No LR warmstarting
- No logger (only to terminal) Maybe use neps's tblogger
- No grad accumulation
"""

import time
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from litgpt.model import GPT, Config
from litgpt.utils import CycleIterator, num_parameters
from torch.utils.data import DataLoader

from scales.data_utils import DataHandler, preprocess_wikitext


def train(
    fabric: L.Fabric,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens_per_step: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_val_steps: int,
    max_seq_length: int,
    max_train_steps: int | None = None,
    max_train_tokens: int | None = None,
    nbr_steps_to_validate: int = 5,
):
    if max_train_steps is not None and max_train_tokens is not None:
        raise ValueError("One of `max_train_steps` or `max_train_tokens` should be set.")
    elif max_train_tokens and max_train_steps:
        raise ValueError("Only one of `max_train_steps` or `max_train_tokens` can be set.")

    # Let's use cycleiterator to do max tokens...
    train_iterator = CycleIterator(train_dataloader)

    tokens_counter = 0
    steps = 0
    for batch in train_iterator:
        if max_train_steps is not None and steps >= max_train_steps:
            break

        # Properly adjust the dimensions
        input_ids = batch[:, 0:max_seq_length].contiguous().long()
        targets = batch[:, 1 : (max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        loss = nn.functional.cross_entropy(logits, targets)

        # update weights
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        tokens_counter += tokens_per_step
        fabric.print(f"Train Step {steps} - Tokens {tokens_counter} - Loss {loss}")

        if steps % nbr_steps_to_validate == 0 or steps == 0:
            validate(
                fabric,
                model,
                val_dataloader,
                max_seq_length,
                max_val_steps,
            )

        steps += 1
        if max_train_tokens is not None and tokens_counter >= max_train_tokens:
            break


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

    val_loss = torch.stack(val_losses).mean()
    fabric.print(f"Validation Loss: {val_loss}")
    model.train()
    fabric.barrier()
    return val_loss


def main(
    fabric: L.Fabric,
    nbr_steps_to_validate: int | None = None,
    max_train_steps: int | None = None,
    max_train_tokens: int | None = None,
    max_val_steps: int | None = None,
    model_name: str | None = None,
    model_config_file: str | Path | None = None,
    seed: int = 1337,
    data: DataHandler = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-2-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
        batch_size=4,
        block_size=32,
    ),
):
    fabric.seed_everything(seed)

    # Setting up model configuration
    if model_config_file and model_name is None:
        config = Config.from_file(model_config_file)
    elif model_name and model_config_file is None:
        config = Config.from_name(model_name)
    elif model_config_file is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    else:
        raise ValueError(f"Please specify model_name")
    model = GPT(config)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    model = fabric.setup(model)

    # setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.02, betas=(0.9, 0.95))
    optimizer = fabric.setup_optimizers(optimizer)

    # Setting up the data with the relevant tokenizer
    data.load_data_laoders()

    train_dataloader = data.data_loaders["train"]
    val_dataloader = data.data_loaders["validation"]

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    fabric.print(f"Steps for training an epoch: {len(train_dataloader)}")
    fabric.print(f"Steps for validation: {len(val_dataloader)}")

    tokens_per_step = data.batch_size * data.block_size

    train_time = time.perf_counter()

    # Training and Validation
    train(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        tokens_per_step=tokens_per_step,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_val_steps=max_val_steps,
        max_seq_length=data.block_size,
        max_train_steps=max_train_steps,
        max_train_tokens=max_train_tokens,
        nbr_steps_to_validate=nbr_steps_to_validate,
    )

    fabric.print(f"Train time: {(time.perf_counter()-train_time):.3f}s")


if __name__ == "__main__":
    # Test run
    fabric = L.Fabric(devices="auto", strategy="auto")
    main(
        fabric=fabric,
        max_train_steps=50,
        max_val_steps=2,
        model_name="pythia-14m",
        nbr_steps_to_validate=10,
    )
