from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import lightning as L
import neps
from jsonargparse import CLI
from neps.plot.tensorboard_eval import tblogger

from scales.config import DataHandler, TrainConfig
from scales.config.ConfigWrapper import ConfigWrapper
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main


def launch_neps(root_dir: Path | str, data_dir: Path | str, seed: int = 449) -> None:
    random.seed(seed)
    if data_dir is not None and isinstance(data_dir, str):
        data_root_path = Path(data_dir)
        assert data_root_path.exists(), f"The root data folder {data_root_path} does not exist"
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
        assert root_dir.parent.exists(), f"The root_dir {root_dir} does not exist"
        assert root_dir.parent.is_dir(), "root_dir must be a directory"

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
    )

    pipeline_space = {
        "lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True, default=1e-3),
        "wd": neps.FloatParameter(lower=1e-5, upper=1e-1, log=False, default=1e-2),
        "warmup_steps": neps.IntegerParameter(lower=0, upper=3000, default=0),
        "eta_min": neps.FloatParameter(lower=0.0, upper=1e-5, log=False, default=0.0),
    }

    def run_pipeline(pipeline_directory: Path, previous_pipeline_directory: Path, **hparams: Any) -> float:
        fabric = L.Fabric(accelerator="auto", strategy="auto")

        train_conf = TrainConfig(
            init_lr=hparams.pop("lr"),
            micro_batch_size=8,
            accumulation_iters=4,
            block_size=1024,
            weight_decay=hparams.pop("wd"),
            max_val_steps=2,
            n_warmup_steps=hparams.pop("warmup_steps"),
            n_main_steps=None,
            n_cooldown_steps=None,
            save_state_every=1000,
            torch_scheduler="CosineAnnealingLR",
            torch_scheduler_args={"T_max": None, "eta_min": hparams.pop("eta_min")},
            model_config=ConfigWrapper(d_model=256, n_layer=12, n_head=2),
            max_train_steps=10000,
            tracked_metrics={
                "train_loss": 1,
                "validation_loss": 5,
                "learning_rate": 1,
                "total_gradient_norm": 10,
                "output_logits_mean": 10,
                "output_logits_max": 10,
                "gradient_norm_per_layer": 20,
            },
            seed=random.randint(0, 100),
        )

        output_dir = pipeline_directory / "output"
        output_dir.mkdir(exist_ok=True)

        config_name = pipeline_directory.name
        train_conf.write_yaml(output_dir / "train_config.yaml", ignore_defaults=False)
        data.write_yaml(output_dir / "data_config.yaml", ignore_defaults=False)

        result_dict = main(fabric=fabric, data=data, train_args=train_conf, out_dir=output_dir)

        tblogger.log(
            loss=result_dict["val_loss"],
            current_epoch=1,
            writer_config_hparam=True,
            write_summary_incumbent=False,
            writer_config_scalar=False,
        )

        return result_dict["val_loss"]

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=root_dir,
        max_evaluations_total=120,
        searcher="random_search",
    )


if __name__ == "__main__":
    CLI(launch_neps)
