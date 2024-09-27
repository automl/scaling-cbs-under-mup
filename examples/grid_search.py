from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import time

import yaml
from scales.config import PipelineConfig
from examples.generate_configs import configs_from_grid
from examples.run_folder import create_slurm_script_folder
from scales.tblog_utils import read_csv_exp_group
from scales.SlurmManager import SlurmManager
import pandas as pd

def get_concat_df(exp_folder: str, hparams: list[str], force_reload: bool = False) -> pd.DataFrame:
    csv_name = "concat_results.csv"
    csv_path = Path(exp_folder + "/" + csv_name)
    if csv_path.exists() and not force_reload:
        df = pd.read_csv(csv_path, index_col=[0, 1], float_precision="round_trip")
    else:
        df = read_csv_exp_group(Path(exp_folder), hparams=hparams, force_reload=force_reload)
        df.to_csv(str(csv_path))
    # Make sure the step index is integer
    df.index = df.index.set_levels(df.index.levels[1].astype(int), level=1)
    return df


def select_best_train(df: pd.DataFrame, last_n=5, quantile=0.1, target_col="Train Loss") -> list[str]:
    grouped_df = df.groupby(level=0)

    group_perf = grouped_df.tail(last_n).groupby(level=0)[target_col].mean()

    sorted_group_perf = group_perf.sort_values(ascending=True)

    threshold = sorted_group_perf.quantile(quantile)

    best_group = sorted_group_perf[sorted_group_perf <= threshold]
    return best_group.index.values

config_grid = {
        "train_config.max_lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "train_config.weight_decay": [1e-5, 3e-5, 1e-4, 3e-4],
        "train_config.adam_beta_2": [0.9, 0.99, 0.999, 0.9999],
        "train_config.adam_beta_1": [0.9, 0.99],
        "train_config.adam_epsilon": [1e-8, 1e-9, 1e-7],
        "train_config.z_loss_eps": [1e-5, 1e-3, 1e-2],
        "train_config.model_config.norm_eps": [1e-7, 1e-5, 1e-3],
        "train_config.model_config.rotary_percentage": [0.25, 0.5, 1.0],
        "train_config.warmup_fraction": [0.001, 0.01, 0.1, 0.05],
        "train_config.cooldown_fraction": [0.01, 0.1, 0.05, 0.2],
        "train_config.weight_init_type": ["plain", "scaled", "GPT-NeoX", "DeepSeek", "none"],
        "train_config.cooldown_type": ["rsqrt", "linear"],
        "train_config.scheduler_type": ["cosine", "linear"],
     }

initial_defaults = {
        "train_config.max_lr": [3e-3],
        "train_config.weight_decay": [3e-5],
        "train_config.adam_beta_2": [0.99],
        "train_config.adam_beta_1": [0.99],
        "train_config.adam_epsilon": [1e-9],
        "train_config.z_loss_eps": [1e-3],
        "train_config.model_config.norm_eps": [1e-5],
        "train_config.model_config.rotary_percentage": [0.5],
        "train_config.warmup_fraction": [0.01],
        "train_config.cooldown_fraction": [0.1],
        "train_config.weight_init_type": ["none"],
        "train_config.cooldown_type": ["rsqrt"],
        "train_config.scheduler_type": ["linear"],
     }

HP_ranking = ["train_config.max_lr",
              "train_config.weight_decay",
              "train_config.adam_beta_2",
              "train_config.adam_beta_1",
              "train_config.adam_epsilon",
              "train_config.warmup_fraction",
              "train_config.cooldown_fraction",
              "train_config.cooldown_type",
              "train_config.weight_init_type",
              "train_config.scheduler_type",
              "train_config.z_loss_eps",
              "train_config.model_config.norm_eps",
              "train_config.model_config.rotary_percentage"
              ]

gs_root_dir = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/gs/scale=0/custom=1")
gs_root_dir.mkdir(exist_ok=True, parents=True)
default_config_path = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/custom=1/def_GS_lr_wd_41M.yaml")
default_config = PipelineConfig.from_path(default_config_path)
best_values_per_hp = defaultdict(list)
hparams = [hp.split(".")[-1] for hp in HP_ranking]

def get_results_path(config_path: Path | str) -> Path:
    return Path(str(Path(config_path).absolute()).replace("/configs/", "/results/"))

def get_rank(hp):
    return HP_ranking.index(hp)
def get_default_config(hp):
    return initial_defaults[hp]
def get_space(hp):
    return config_grid[hp]
def get_hp_by_rank(rank):
    return HP_ranking[rank]

def collect_results(configs, main_rank: int, secondary_rank: int | None = None, sub_grid: dict | None = None) -> dict:
    # Write Configs
    secondary_rank = secondary_rank if secondary_rank is not None else "none"
    configs_folder = gs_root_dir / f"hp_{main_rank}/hp_{secondary_rank}"
    configs_folder.mkdir(exist_ok=True, parents=True)

    config_prefix = "pipeline"
    _ = [
        config.write_yaml(output_dir=configs_folder / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(configs)
    ]
    # End Write Configs

    #Write Subgrid
    with (configs_folder / "sub_grid.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(sub_grid, stream)

    # Submit Configs
    results_folder = get_results_path(configs_folder)
    slurm_manager = SlurmManager(experiment_name=f"GS_hp_{main_rank}_hp_{secondary_rank}"
                                 )
    slurm_manager.create_slurm_script_folder(partition="bosch_gpu-rtx2080",
                                             max_time="0-12:00",
                                             gpu_per_job=default_config.train_config.devices,
                                             root_output_path=str(results_folder),
                                             root_configs_path=str(configs_folder))
    if not slurm_manager.check_run_completed():
        slurm_manager.submit()
    # End Submit

    # Wait for completion
    waiting_time = 0
    while not slurm_manager.check_run_completed():
        print(f"Waited {waiting_time} hours")
        print("Waiting...")
        time.sleep(3600)
        waiting_time += 1

    # Collect Results
    print(f"All configs completed in {waiting_time} hours\nCollecting results...")
    concat_df = get_concat_df(str(results_folder), hparams)
    best_configs = select_best_train(concat_df)
    best_config =  concat_df.loc[(best_configs[0], concat_df.loc[best_configs[0]].index.max())].to_dict()

    # Write the Best Config
    with (results_folder / "best_configs.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(best_configs, stream)
    return best_config


def get_subgrid(target_hp):
    sub_grid = {}

    for key in config_grid.keys():
        if key == target_hp:
            sub_grid[key] = get_space(key)
        elif best_values_per_hp[key]:
            sub_grid[key] = best_values_per_hp[key][-1]
        else:
            sub_grid[key] = get_default_config(key)
    return sub_grid


for idx, hp in enumerate(HP_ranking):
    sub_grid = get_subgrid(hp)

    configs = configs_from_grid(default_config, sub_grid)
    best_config = collect_results(configs, idx, sub_grid=sub_grid)
    # best_config = get_best_config(sub_grid, results)
    best_values_per_hp[hp].append(best_config[hp])

    for i in range(idx):
        sub_grid = get_subgrid(get_hp_by_rank(i))

        configs = configs_from_grid(default_config, sub_grid)
        best_config = collect_results(configs, idx, i, sub_grid=sub_grid)
        # best_config = get_best_config(sub_grid, results)
        best_values_per_hp[get_hp_by_rank(i)].append(best_config[get_hp_by_rank(i)])

print({hp:best_values_per_hp[hp][-1] for hp in HP_ranking})