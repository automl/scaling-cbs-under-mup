import logging
from pathlib import Path
from typing import Any

import lightning as L
import neps
from litgpt.config import Config

from scales.config.data_config import DataHandler
from scales.config.train_config import PipelineConfig, TrainConfig
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main


def search_space() -> dict:
    return {
        "lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True, default=1e-3),
    }


def run_pipeline(pipeline_directory: Path, previous_pipeline_directory: Path, **hparams: Any) -> float:
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

    train_conf = TrainConfig(
        init_lr=hparams.get("lr"),  # type: ignore
        micro_batch_size=1,
        block_size=1024,
        weight_decay=0.0,
        max_val_steps=2,
        accumulation_iters=1,
        n_warmup_steps=None,
        n_main_steps=None,
        n_cooldown_steps=None,
        torch_scheduler="CosineAnnealingLR",
        torch_scheduler_args={"T_max": None, "eta_min": 5e-4},
        model_config=Config(block_size=1024, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=32),
        tracked_metrics={
            "train_loss": 5,
            "validation_loss": 5,
            "total_gradient_norm": 10,
            "output_logits_mean": 10,
            "output_logits_max": 10,
            "gradient_norm_per_layer": 20,
            "max_attention_logits_per_layer": 10,
            "max_attention_logits_all": 10,
        },
        max_train_steps=10000,
    )

    config = PipelineConfig(data_config=data_handler, train_config=train_conf)

    data_handler = config.data_config

    result_dict = main(
        fabric=fabric,
        data=data_handler,  # type: ignore
        train_args=config.train_config,  # type: ignore
        out_dir=pipeline_directory,
    )

    return result_dict["val_loss"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=search_space(),
        root_directory=Path(__file__).parent / "tune_lr",
        max_evaluations_total=20,
        searcher="bayesian_optimization",
    )
