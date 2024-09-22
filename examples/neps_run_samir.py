from __future__ import annotations

import os
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from dataclasses import fields
import neps
import yaml
from jsonargparse import CLI

from scales.config import DataHandler, TrainConfig
from scales.config.ConfigWrapper import ConfigWrapper
from neps.plot.tensorboard_eval import tblogger
from scales.neps_utils import DEFAULT_TRAIN_CONFIG_FILE, DEFAULT_DATA_CONFIG_FILE, dynamic_micro_batch

DEFAULT_BATCH_SIZE = 64


DEFAULTS_TRAIN_CONFIG = TrainConfig(
    max_lr=1e-3,
    micro_batch_size=1,
    block_size=1024,
    weight_decay=1e-2,
    max_val_steps=5,
    accumulation_iters=1,
    devices="auto",

    model_config=ConfigWrapper(d_model=64, n_head=4, n_layer=2, bias=True, lm_head_bias=False),
    # weight_init_type=None,

    cooldown_fraction=0.2,
    warmup_fraction=0.05,
    cooldown_type="rsqrt",
    min_lr=.0,
    torch_scheduler=None,  #None=constant
    torch_scheduler_args=None,

    tokens_per_param=20,
    # max_train_steps=100,
    clip_max_val=5.0,
    validate_every=5,
    z_loss_eps=1e-4,

    adam_beta_1=0.9,
    adam_beta_2=0.95,
    adam_eps=1e-8,
    independent_wd=True,

    mup_base_scales=None,
    save_state_every=4000,

    tracked_metrics={"learning_rate": 1,
                     "train_loss": 1,
                     "output_logits_max": 10,
                     "output_logits_mean": 10,
                     "max_attention_logits_per_layer": 10,
                     "max_attention_logits_all": 10,
                     "optimizer_stats": 10,
                     "total_gradient_norm": 20,
                     "gradient_norm_per_layer": 20,
                     "validation_loss": 5,
                     "weight_spectra_max": 5,
                     "weight_spectra_diff": 5
                     },

    seed=444,
)

pipeline_config = {
    # An example pipeline space
    "max_lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True, default=1e-3, default_confidence="low"),
    "weight_decay": neps.FloatParameter(lower=1e-5, upper=1e-1, log=False, default=1e-2, default_confidence="medium"),
    "adam_beta_2": neps.FloatParameter(lower=0.9, upper=0.999, log=True, default=0.995, default_confidence="medium"),
    "adam_beta_1": neps.FloatParameter(lower=0.9, upper=0.99, log=True, default=0.9, default_confidence="medium"),
    "z_loss_eps": neps.FloatParameter(lower=1e-5, upper=1e-2, log=True, default=1e-4, default_confidence="high"),

    # "width_to_depth": neps.IntegerParameter(lower=16, upper=128, default=32),
    # "d_model_scale": neps.IntegerParameter(lower=1, upper=8, is_fidelity=True),
    "head_size_scale": neps.IntegerParameter(lower=1, upper=6, is_fidelity=True),
    "n_head_scale": neps.IntegerParameter(lower=1, upper=4, default=1, default_confidence="low"),
    "n_layer": neps.IntegerParameter(lower=4, upper=16, default=4, default_confidence="medium"),
}

hpo_config = {
    "max_lr": neps.FloatParameter(lower=1e-5, upper=1e-1, log=True, default=1e-3, default_confidence="medium"),
    "weight_decay": neps.FloatParameter(lower=1e-5, upper=1e-1, log=False, default=1e-2, default_confidence="medium"),
    "adam_beta_2": neps.FloatParameter(lower=0.9, upper=0.999, log=True, default=0.995, default_confidence="medium"),
    "adam_beta_1": neps.FloatParameter(lower=0.9, upper=0.99, log=True, default=0.9, default_confidence="medium"),
    "adam_eps": neps.FloatParameter(lower=1e-9, upper=1e-7, log=True, default=1e-8, default_confidence="high"),
    "z_loss_eps": neps.FloatParameter(lower=1e-5, upper=1e-2, log=True, default=1e-4, default_confidence="high"),
    "norm_eps": neps.FloatParameter(lower=1e-7, upper=1e-3, log=True, default=1e-5, default_confidence="high"),

    "rotary_percentage": neps.CategoricalParameter(choices=[0.25, 0.50, 1.0], default=0.25, default_confidence="high"),

    "cooldown_type": neps.CategoricalParameter(choices=["rsqrt", "linear"], default="rsqrt", default_confidence="high"),
    "scheduler_type": neps.CategoricalParameter(choices=["cosine", "constant"], default="cosine", default_confidence="high"),
    "weight_init_type": neps.CategoricalParameter(choices=["plain", "scaled", "GPT-NeoX", "DeepSeek", "none"], default="GPT-NeoX", default_confidence="high"),

    "min_lr_percentage": neps.FloatParameter(lower=0.0, upper=0.1, default=0.1, default_confidence="medium"),
    "warmup_fraction": neps.FloatParameter(lower=0.001, upper=0.1, default=0.05, default_confidence="medium"),
    "cooldown_fraction": neps.FloatParameter(lower=0.0, upper=0.3, default=0.2, default_confidence="high"),
    # block_size = 1024
    "max_train_steps": neps.IntegerParameter(lower=1000, upper=21000, is_fidelity=True),

}

def prep_train_config_hpo(**hparams) -> TrainConfig:
    """
    This function defines how to create a TrainConfig from the hpo space.
    It generates a config with only one scale for the model, and replaces all the HPs with the given values.
    """
    train_conf_fields = [field.name for field in fields(TrainConfig)]
    train_conf_params = {}
    other_params = {}
    for name, val in hparams.items():
        if name in train_conf_fields:
            train_conf_params[name] = val
        else:
            other_params[name] = val
    train_conf_dict = DEFAULTS_TRAIN_CONFIG.to_dict()
    # Ensure max_train_steps is used instead of tokens_per_param
    # NOTE: tokens_per_param has precendence over max_train_steps
    train_conf_dict["tokens_per_param"] = None
    DEFAULT_BATCH_SIZE = 64
    target_scale = {"d_model": 384, "n_head": 4, "n_layer": 16}

    target_config = ConfigWrapper(apply_qk_norm=True, 
                                  rotary_percentage=other_params["rotary_percentage"], 
                                  norm_eps=other_params["norm_eps"],
                                  **target_scale,)
    
    min_lr = train_conf_params["max_lr"] * other_params["min_lr_percentage"]
    update_params = {"model_config": target_config.to_dict(ignore_defaults=True), 
                     "min_lr": min_lr,
                     "torch_scheduler": "CosineAnnealingLR" if other_params["scheduler_type"] == "cosine" else None,
                    #  to make sure that cosine decay doesn't reach min_lr before cooldown starts
                     "torch_scheduler_args": {"eta_min": min_lr if train_conf_params["cooldown_fraction"] > 0 else min_lr * 2,},
                     "mup_base_scales": None,}

    train_conf_params.update(update_params)
    train_conf_dict.update(train_conf_params)

    return TrainConfig.from_yaml(yaml_config=train_conf_dict)


def prep_train_config_1(**hparams) -> TrainConfig:
    """
    This function defines how to create a TrainConfig from a given pipeline space.
    Therefore, for every unique pipeline space there should be a corresponding prep_train_config function.
    Make sure to adjust all the constant values you want to change here, instead of changing the DEFAULTS_TRAIN_CONFIG
    Alternatively, you can define your own defaults separately.

    Args:
        hparams: neps config dict
    Returns:
        Initialized TrainConfig object
    """
    train_conf_fields = [field.name for field in fields(TrainConfig)]
    train_conf_params = {}
    other_params = {}
    for name, val in hparams.items():
        if name in train_conf_fields:
            train_conf_params[name] = val
        else:
            other_params[name] = val

    train_conf_dict = DEFAULTS_TRAIN_CONFIG.to_dict()

    mup_base_scales = {"d_model": 16, "n_head": 2}

    # when initialized model will have impossible config e.g. n_layer < 1
    # if int(mup_base_scales["d_model"] * other_params["d_model_scale"] / other_params["width_to_depth"]) < 1:
    #     raise ValueError("The configuration is not possible")
    base_head_size = 8
    head_size = base_head_size * other_params["head_size_scale"]
    n_head = mup_base_scales["n_head"] * other_params["n_head_scale"]

    config_wrapper = ConfigWrapper(d_model=head_size * n_head,
                                   n_head=mup_base_scales["n_head"] * other_params["n_head_scale"],
                                   n_layer=other_params["n_layer"],
                                   apply_qk_norm=True,)

    update_params = {"mup_base_scales": mup_base_scales,
                     "model_config": config_wrapper.to_dict(ignore_defaults=True),}

    train_conf_params.update(update_params)
    train_conf_dict.update(train_conf_params)

    return TrainConfig.from_yaml(yaml_config=train_conf_dict)


# =====================================================================
# ===============Below is almost experiment invariant==================
# =====================================================================
# Remember to add your experiment's details to the neps_experiment_dict


neps_experiment_dict = {"scale_mup": {"pipeline_space": pipeline_config,
                                      "prep_train_config": prep_train_config_1,
                                      "api_kwargs": {"max_evaluations_total": 60,
                                                     "searcher": "priorband"}
                                      },
                        "hpo_sp": {"pipeline_space": hpo_config,
                                      "prep_train_config": prep_train_config_hpo,
                                      "api_kwargs": {"max_evaluations_total": 60,
                                                     "searcher": "priorband"}
                                      },
                        }


# From neps examples
def _ask_to_submit_slurm_script(pipeline_directory: Path, script: str) -> None:
    script_path = pipeline_directory / "submit.sh"
    logging.info(f"Submitting the script {script_path} (see below): \n\n{script}")

    # You may want to remove the below check and not ask before submitting every time
    # if input("Ok to submit? [Y|n] -- ").lower() in {"y", ""}:
    script_path.write_text(script)
    os.system(f"sbatch {script_path}")
    # else:
    #     raise ValueError("We generated a slurm script that should not be submitted.")


def _get_result_dict(pipeline_directory: Path) -> dict | None:
    result_file = pipeline_directory / "output" / "result.yaml"
    finished_file = pipeline_directory / "slurm_finished.txt"
    # Get the last modified output log
    
    if finished_file.exists() and result_file.exists():
        with result_file.open(encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)
    return None

def decide_halt(pipeline_directory: Path) -> bool:
    log_dir = pipeline_directory / "log"
    files = [file for file in log_dir.iterdir() if file.is_file() and file.suffix == ".out"]
    files.sort(key=lambda x: x.stat().st_mtime)
    last_line = None
    job_id = None
    if files:
        log_file = files[-1]
        job_id = log_file.name.split(".")[2]
        with log_file.open("r", encoding="utf-8") as log_file:
            # get the last two lines of the log file
            last_line = "".join(log_file.readlines()[-2:])
    
    return last_line, job_id
    


def get_run_pipeline(data, prep_train_config: Callable[[dict], TrainConfig], loss_on_init_error: float = 10., device_count = 1) -> Callable:
    def generic_run_pipeline(pipeline_directory: Path, previous_pipeline_directory: Path, **hparams: Any) -> float:
        """
        This function will handle the input configs and create a train_config and data_config yaml files
        for the slurm to read from, then it'll call the slurm script to run the training and wait until the job is finished.
        """

        # TODO: Handle continuations from previous_pipeline_directory

        try:
            train_config = prep_train_config(**hparams)
        except ValueError as e:
            # Do something when it's not possible to init the specified model
            if "not possible" in str(e):
                return loss_on_init_error
            else:
                raise e

        output_dir = pipeline_directory / "output"
        output_dir.mkdir(exist_ok=True)

        logdir = pipeline_directory / "log"
        logdir.mkdir(exist_ok=True) 

        script = f"""#!/bin/bash
#SBATCH -c {device_count}
#SBATCH --time 0-24:00
#SBATCH --job-name test
#SBATCH --partition mlhiwidlc_gpu-rtx2080
#SBATCH --error "{logdir}/%x.%A.%a.%N.err"
#SBATCH --gres=gpu:{device_count}
#SBATCH --ntasks-per-node={device_count}
#SBATCH --output "{logdir}/%x.%A.%a.%N.out"

# Debug info
export NCCL_DEBUG=DEBUG
export PYTHONFAULTHANDLER=1
# read yaml files and train here
srun poetry run python -c "from scales.neps_utils import dynamic_micro_batch; dynamic_micro_batch(r'{output_dir}')"
echo finished > {pipeline_directory}/slurm_finished.txt
"""

        data.write_yaml(output_dir / DEFAULT_DATA_CONFIG_FILE, ignore_defaults=False)

        train_config.micro_batch_size = DEFAULT_BATCH_SIZE // device_count
        train_config.devices = device_count
        train_config.accumulation_iters = 1
        if previous_pipeline_directory is not None:
            checkpoint_dir = previous_pipeline_directory / "output"
            train_config.load_state_path = checkpoint_dir
        train_config.write_yaml(output_dir / DEFAULT_TRAIN_CONFIG_FILE, ignore_defaults=True)
        # TODO: Is there any way to submit jobs without occupying the login node?
        # result_dict = dynamic_micro_batch(output_dir)
        _ask_to_submit_slurm_script(pipeline_directory, script)
        result_dict = None
        previous_last_line = "Not started yet"
        halt_counter = 0
        while result_dict is None:
            logging.info("Waiting until the job has finished.")
            time.sleep(3600)  # 1h
            result_dict = _get_result_dict(pipeline_directory)
            last_line, job_id = decide_halt(pipeline_directory)
            if last_line != previous_last_line:
                logging.info(f"Last line of the log file: {last_line}")
                previous_last_line = last_line
            else:
                logging.info("Halting detected. Re-submitting the job.")
                logging.info(f"Cancelling the job: {job_id}")
                os.system(f"scancel --signal=SIGINT {job_id}")
                # Move the output directory to a new one
                previous_checkpoint = output_dir.rename(output_dir.parent / f"{output_dir.name}_halt_{halt_counter}")
                output_dir.mkdir(exist_ok=True)
                data.write_yaml(output_dir / DEFAULT_DATA_CONFIG_FILE, ignore_defaults=False)
                # load from the halted checkpoint
                train_config.load_state_path = previous_checkpoint
                train_config.write_yaml(output_dir / DEFAULT_TRAIN_CONFIG_FILE, ignore_defaults=True)
                halt_counter += 1
                _ask_to_submit_slurm_script(pipeline_directory, script)

        tblogger.log(loss=result_dict["val_loss"],
                     current_epoch=1,
                     writer_config_hparam=True,
                     write_summary_incumbent=False,
                     writer_config_scalar=False)

        return result_dict["val_loss"]

    return generic_run_pipeline


def launch_neps(root_dir: Path | str, dataset_dir: Path | str, experiment_name: str, device_count: int = 2, seed: int = 449) -> None:
    random.seed(seed)
    if dataset_dir is not None and isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
        assert (dataset_dir / "DataHandler.yaml").exists(), f"DataHandler.yaml doesn't exist at {dataset_dir}"
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
        root_dir.mkdir(exist_ok=True)

    root_dir = root_dir / experiment_name
    # root_dir.mkdir(exist_ok=True, parents=True)

    data = DataHandler.from_path(dataset_dir)

    pipeline_space = neps_experiment_dict[experiment_name]["pipeline_space"]
    run_pipeline = get_run_pipeline(data=data,
                                    prep_train_config=neps_experiment_dict[experiment_name]["prep_train_config"],
                                    device_count=device_count)

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=root_dir,
        **neps_experiment_dict[experiment_name]["api_kwargs"]
    )


if __name__ == "__main__":
    CLI(launch_neps)
