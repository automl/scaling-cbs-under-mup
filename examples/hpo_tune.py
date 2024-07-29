import argparse
import logging
import random
from pathlib import Path
from typing import Any

import lightning as L
import neps
import neps.api
import numpy as np
import torch
from litgpt.config import Config

from scales.config.data_config import DataHandler
from scales.config.train_config import PipelineConfig, TrainConfig
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main

SEED = 444


def search_space(accumulation_iters: int = 1) -> dict:
    return {
        "lr_exponent": neps.FloatParameter(lower=1, upper=5),
        "accumulation_iters": neps.ConstantParameter(value=accumulation_iters),
    }


def run_pipeline(pipeline_directory: Path, **hparams: Any) -> dict[str, Any]:
    fabric = L.Fabric(accelerator="auto", strategy="auto")

    data_handler = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=True,
        subsample_index=0,
    )

    lr_exponent = -float(hparams.get("lr_exponent"))  # type: ignore

    train_conf = TrainConfig(
        max_lr=10**lr_exponent,
        seed=SEED,
        weight_init_type="GPT-NeoX",
        micro_batch_size=8,
        block_size=1024,
        weight_decay=0.0,
        max_val_steps=2,
        accumulation_iters=hparams.get("accumulation_iters"),  # type: ignore
        model_config=Config(block_size=1024, n_layer=24, n_head=4, vocab_size=50257, bias=True, n_embd=96),
        tracked_metrics={
            "train_loss": 1,
            "learning_rate": 5,
            "optimizer_stats": 1,
            "validation_loss": 5,
            "total_gradient_norm": 10,
            "output_logits_mean": 10,
            "output_logits_max": 10,
            "gradient_norm_per_layer": 20,
            "max_attention_logits_per_layer": 5,
            "max_attention_logits_all": 5,
        },
        tokens_per_param=20,
    )

    config = PipelineConfig(data_config=data_handler, train_config=train_conf)

    data_handler = config.data_config

    result_dict = main(
        fabric=fabric,
        data=data_handler,  # type: ignore
        train_args=config.train_config,  # type: ignore
        out_dir=pipeline_directory / "gpt_output",
    )

    return {
        "loss": result_dict["train_loss"],
        "lr": 10**lr_exponent,
        "val_loss": result_dict["val_loss"],
    }


def set_seed(seed: int = 123) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for SP training")
    parser.add_argument(
        "--accumulation_iters", type=int, default=1, help="Accumulation iterations for effective batch size"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(SEED)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=search_space(accumulation_iters=args.accumulation_iters),
        root_directory=Path(__file__).parent / f"tune_lr_accit{args.accumulation_iters}",
        max_evaluations_total=10,
        searcher="grid_search",
        grid_step_size=10,
    )
