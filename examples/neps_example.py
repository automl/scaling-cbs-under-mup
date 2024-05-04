import logging
from pathlib import Path
from typing import Any

import lightning as L
import neps

from scales.data_utils import DataHandler, preprocess_wikitext
from scales.pretrain_pipeline import main

pipeline_space = {
    "lr": neps.FloatParameter(lower=1e-5, upper=5e-3, log=True, default=1e-3),
    "wd": neps.FloatParameter(lower=1e-5, upper=5e-3, log=True, default=1e-4),
    "steps": neps.IntegerParameter(lower=2, upper=10, is_fidelity=True),
}


def run_pipeline(pipeline_directory: Path, previous_pipeline_directory: Path, **hparams: Any) -> float:
    fabric = L.Fabric(accelerator="auto", strategy="auto")

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-2-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
    )

    if previous_pipeline_directory is None:
        result_dict = main(
            fabric=fabric,
            data=data,
            out_dir=pipeline_directory / "output",
            hparams={"lr": hparams["lr"], "weight_decay": hparams["wd"], "batch_size": 64, "block_size": 2048},
            model_config_file="model.yaml",
            max_train_steps=hparams["steps"],
            max_val_steps=2,
        )
    else:
        result_dict = main(
            fabric=fabric,
            data=data,
            out_dir=pipeline_directory / "output",
            hparams={"lr": hparams["lr"], "weight_decay": hparams["wd"], "batch_size": 64, "block_size": 2048},
            load_model_from_path=previous_pipeline_directory / "output",
            max_train_steps=hparams["steps"],
            max_val_steps=2,
        )

    return result_dict["val_loss"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="result",
        max_evaluations_total=10,
        searcher="priorband",
    )
