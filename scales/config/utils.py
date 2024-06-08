from __future__ import annotations

import re
from pathlib import Path

import torch
from datasets import Dataset
from litgpt.tokenizer import Tokenizer


def download_tokenizer(repo_id: str, root_dir: str | Path, overwrite: bool = False) -> None:
    """Download the trained tokenizer from the selected HF repo. The tokwenizer will be saved under /root_dir/repo_id.

    Note: To use HF token for authentication set the environment variable HF_TOKEN

    Args:
    ----
    repo_id: HuggingFace repository id for the tokenizer to be downloaded from
    root_dir: path to save the tokenizer under

    """
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    if (root_dir / repo_id).exists() and not overwrite:
        # Tokenizer already exists
        return
    from litgpt.scripts.download import download_from_hub

    download_from_hub(repo_id=repo_id, tokenizer_only=True, checkpoint_dir=root_dir, convert_checkpoint=False)


def preprocess_wikitext(row: dict[str, str]) -> dict:
    """Wikitext specific preprocessing, removes new_line at the end of each file."""
    row["text"] = re.sub(r"\n$", "", row["text"])
    return row


def tokenize_wikitext(indices: list[int] | int, dataset: Dataset, tokenizer: Tokenizer) -> torch.Tensor:
    """Tokenize each row in the dataset into tensors."""
    # only yield for now due to a bug on litdata
    # https://github.com/Lightning-AI/litdata/issues/70
    if isinstance(indices, int):
        yield tokenizer.encode(dataset[indices]["text"], eos=True)
    else:
        # Yield batches to be fast on large datasets
        yield torch.cat([tokenizer.encode(dataset[index]["text"], eos=True) for index in indices])


def simple_filter(row: dict) -> bool:
    return len(row["text"]) > 1
