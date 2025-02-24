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


def search_space(
    accumulation_iters: int = 1,
    width: int = 96,
    micro_batch_size: int = 8,
    n_gpus: int = 1,
    total_validation_tokens: int = 2500000,
) -> dict:
    return {
        "lr_exponent": neps.FloatParameter(lower=1, upper=5),
        "accumulation_iters": neps.ConstantParameter(value=accumulation_iters),
        "width": neps.ConstantParameter(value=width),
        "micro_batch_size": neps.ConstantParameter(value=micro_batch_size),
        "n_gpus": neps.ConstantParameter(value=n_gpus),
        "total_validation_tokens": neps.ConstantParameter(value=total_validation_tokens),
    }


def run_pipeline(pipeline_directory: Path, **hparams: Any) -> dict[str, Any]:
    fabric = L.Fabric(accelerator="auto", strategy="auto")

    total_validation_steps = int(
        hparams.get("total_validation_tokens")
        / (hparams.get("micro_batch_size") * hparams.get("n_gpus") * hparams.get("block_size"))
    )

    data_handler = DataHandler(
        hf_dataset_id="DKYoon/SlimPajama-6B",
        hf_data_subset_name="",
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
        micro_batch_size=hparams.get("micro_batch_size"),
        block_size=1024,
        weight_decay=0.0,
        max_val_steps=total_validation_steps,
        validate_every=20,
        accumulation_iters=hparams.get("accumulation_iters"),  # type: ignore
        model_config=Config(
            block_size=1024, n_layer=8, n_head=4, vocab_size=50257, bias=True, n_embd=hparams.get("width")
        ),
        tracked_metrics={
            "train_loss": 1,
            "learning_rate": 5,
            "optimizer_stats": 1,
            "validation_loss": 20,
            "total_gradient_norm": 5,
            "output_logits_mean": 10,
            "output_logits_max": 10,
            "gradient_norm_per_layer": 20,
            "max_attention_logits_per_layer": 5,
            "max_attention_logits_all": 5,
            "tokens_per_step": 1,
            "activations_train": 1,
            "layerwise_features_rms_val": 1,
            "layerwise_features_l1_mean_val": 1,
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
        "effective_batch_size": hparams.get("n_gpus")
        * hparams.get("micro_batch_size")
        * hparams.get("accumulation_iters"),
    }


def set_seed(seed: int = 444) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for SP training")
    parser.add_argument(
        "--accumulation_iters", type=int, default=1, help="Accumulation iterations for effective batch size"
    )
    parser.add_argument("--width", type=int, default=96, help="Width of the GPT model")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="micro batch size fit according to GPU")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs used in this setup")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(SEED)
    effective_batch_size = args.accumulation_iters * args.n_gpus * args.micro_batch_size
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=search_space(
            accumulation_iters=args.accumulation_iters,
            width=args.width,
            micro_batch_size=args.micro_batch_size,
            n_gpus=args.n_gpus,
        ),
        root_directory=Path(__file__).parent / f"results/tune_lr_bs{effective_batch_size}_w{args.width}",
        max_evaluations_total=15,
        searcher="grid_search",
        grid_step_size=15,
    )
