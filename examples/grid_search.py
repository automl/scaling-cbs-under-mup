from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import time
import itertools
from typing import Any
from copy import deepcopy

from jsonargparse import CLI

import yaml
from scales.config import PipelineConfig
from scales.tblog_utils import read_csv_exp_group
from scales.exp_utils.SlurmManager import SlurmManager
import pandas as pd

# from https://stackoverflow.com/a/52099238/8889365
def deep_merge(d: dict, u: dict) -> None:
    """Do a deep merge of one dict into another.

    This will update d with values in u, but will not delete keys in d
    not found in u at some arbitrary depth of d. That is, u is deeply
    merged into d.

    Args -
      d, u: dicts

    Note: this is destructive to d, but not u.

    Returns: None

    """
    stack = [(d, u)]
    while stack:
        d, u = stack.pop(0)
        for k, v in u.items():
            if not isinstance(v, dict):
                # u[k] is not a dict, nothing to merge, so just set it,
                # regardless if d[k] *was* a dict
                d[k] = v
            else:
                # note: u[k] is a dict
                if k not in d:
                    # add new key into d
                    d[k] = v
                elif not isinstance(d[k], dict):
                    # d[k] is not a dict, so just set it to u[k],
                    # overriding whatever it was
                    d[k] = v
                else:
                    # both d[k] and u[k] are dicts, push them on the stack
                    # to merge
                    stack.append((d[k], v))


def key_mapping(key_str: str, value: Any) -> dict[str, Any]:
    keys = key_str.split(sep=".")
    for key in reversed(keys):
        value = {key: value}
    return value


def configs_from_grid(
    default_config: PipelineConfig, config_grid: dict[str, list[str | int | float]]
) -> list[PipelineConfig]:
    config_list = []
    keys = list(config_grid.keys())
    configs_product = itertools.product(*list(config_grid.values()))
    values_list = [dict(zip(keys, values)) for values in configs_product]
    for config in values_list:
        config_dict = deepcopy(default_config.to_dict())
        for key_str, value in config.items():
            new_mapping = key_mapping(key_str, value)
            deep_merge(config_dict, new_mapping)

        config_list.append(PipelineConfig.from_yaml(config_dict))
    return config_list

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

def get_results_path(configs_folder: Path) -> Path:
    return Path(str(configs_folder.absolute()).replace("/configs/", "/results/"))

def write_configs(configs, gs_root_dir: Path, subspace_prefix: str) -> Path:
    configs_folder = gs_root_dir / subspace_prefix
    configs_folder.mkdir(exist_ok=True, parents=True)

    config_prefix = "pipeline"
    _ = [
        config.write_yaml(output_dir=configs_folder / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(configs)
    ]
    return configs_folder

def submit_configs(configs_folder: Path, sub_grid: dict | None = None, 
                   subspace_prefix: str | None = None, 
                   max_time: str = "0-3:00",
                   gpu_per_job: int = 1) -> None:
    results_folder = get_results_path(configs_folder)
    results_folder.mkdir(exist_ok=True, parents=True)
    with (results_folder / "sub_grid.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(sub_grid, stream)

    slurm_manager = SlurmManager(experiment_name=f"GS_{subspace_prefix}",
                                 )
    slurm_manager.create_slurm_script_folder(partition="bosch_gpu-rtx2080",
                                             max_time=max_time,
                                             gpu_per_job=gpu_per_job,
                                             root_output_path=str(results_folder),
                                             root_configs_path=str(configs_folder))
    if not slurm_manager.check_run_completed():
        slurm_manager.submit()
    
    return slurm_manager

def wait_for_completion(slurm_manager: SlurmManager) -> None:
    waiting_time = 0
    while not slurm_manager.check_run_completed():
        print(f"Waited {waiting_time} hours")
        print("Waiting...")
        time.sleep(3600)
        waiting_time += 1
    print(f"All configs completed in {waiting_time} hours\nCollecting results...")

def collect_results(configs_folder: Path, 
                    sub_grid: dict[str, list[int | float | str]], 
                    force_reload: bool = False) -> dict:
    hparams = [hp.split(".")[-1] for hp in sub_grid.keys()]
    results_folder = get_results_path(configs_folder)
    concat_df = get_concat_df(str(results_folder), hparams, force_reload=force_reload)
    best_configs = select_best_train(concat_df)
    best_config =  concat_df.loc[(best_configs[0], concat_df.loc[best_configs[0]].index.max())].to_dict()

    # Write the Best Config
    with (results_folder / "best_configs.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(best_config, stream)
    return best_config

config_grid_example = {"train_config.max_lr": 
                            {"values": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                            "default": 1e-2,
                            "rank": 0},
                      "train_config.weight_decay": 
                            {"values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
                            "default": 1e-3,
                            "rank": 1},
                     "train_config.adam_beta_2": 
                            {"values": [0.9, 0.99, 0.995, 0.999, 0.9999],
                            "default": 0.999,
                            "rank": 2},
                    "train_config.adam_beta_1": 
                            {"values": [0.9, 0.99],
                            "default": 0.99,
                            "rank": 3},
                    "train_config.adam_eps": 
                            {"values": [1e-10, 1e-9, 1e-8, 1e-7],
                            "default": 1e-8,
                            "rank": 4},
                    "train_config.z_loss_eps": 
                            {"values": [1e-5, 1e-3, 1e-2],
                            "default": 1e-3,
                            "rank": 5},
                    "train_config.warmup_fraction": 
                            {"values": [0.001, 0.01, 0.1, 0.05],
                            "default": 0.01,
                            "rank": 6},
                    "train_config.cooldown_fraction": 
                            {"values": [0.01, 0.1, 0.05, 0.2, 0.3, 0.5],
                            "default": 0.1,
                            "rank": 7},
                    "train_config.weight_init_type": 
                            {"values": ["plain", "scaled", "GPT-NeoX", "DeepSeek", "none"],
                            "default": "none",
                            "rank": 8},
                    "train_config.cooldown_type": 
                            {"values": ["rsqrt", "linear"],
                            "default": "rsqrt",
                            "rank": 9},
                    "train_config.torch_scheduler": 
                            {"values": ["cosine", "linear"],
                            "default": "linear",
                            "rank": 10},               
}

def load_config_grid(config_grid_path: Path) -> dict[str, dict[str, list[int | float | str] | int | float | str]]:
    with config_grid_path.open("r", encoding="utf-8") as stream:
        config_grid = yaml.safe_load(stream)
    return config_grid

def save_config_grid(config_grid: dict[str, dict[str, list[int | float | str] | int | float | str]], config_grid_path: Path) -> None:
    with config_grid_path.open("w", encoding="utf-8") as stream:
        yaml.dump(config_grid, stream)


def sub_grid_search(gs_root_dir: Path | str,
                    config_grid: dict[str, 
                                      dict[str, 
                                           list[int | float | str] | int | float | str]] | None = None, 
                    default_config: PipelineConfig | Path | str | None = None, 
                    max_time: str = "0-12:00",
                    repeat_top: int | None = None,
                    waiting_dimension: int | None = None,
                    ) -> dict[str, list]:
    gs_root_dir = Path(gs_root_dir)
    best_values_per_hp = defaultdict(list)

    # Set repeat_top to the length of the config_grid if not provided
    repeat_top = repeat_top if repeat_top is not None else len(config_grid)
    waiting_dimension = waiting_dimension if waiting_dimension is not None else 2
    # Load or save config_grid
    if config_grid is None:
        try:
            # Load config_grid from root_dir
            config_grid = load_config_grid(gs_root_dir / "config_grid.yaml")
        except FileNotFoundError as e:
            raise FileNotFoundError("config_grid.yaml not found. Please provide a config_grid") from e
    else:
        save_config_grid(config_grid, gs_root_dir / "config_grid.yaml")
    # Load or save default_config
    if isinstance(default_config, (Path, str)):
        default_config = PipelineConfig.from_path(Path(default_config))
    elif default_config is None:
        try:
            # Load default_config from root_dir
            default_config = PipelineConfig.from_path(gs_root_dir / "default_config.yaml")
        except FileNotFoundError as e:
            raise FileNotFoundError("default_config.yaml not found. Please provide a default_config") from e
    else:
        default_config.write_yaml(gs_root_dir / "default_config.yaml")

    def get_rank(hp) -> int:
        return config_grid[hp]["rank"]
    def get_default_config(hp) -> int | float | str:
        return config_grid[hp]["default"]
    def get_space(hp) -> list[int | float | str]:
        return config_grid[hp]["values"]
    def get_hp_by_rank(rank) -> str:
        for hp in config_grid.keys():
            if config_grid[hp]["rank"] == rank:
                return hp
    def get_hp_name(hp) -> str:
        return hp.split(".")[-1]
            
    def get_subgrid(target_hp, ignore_default=False):
        sub_grid = {}
        for key in config_grid.keys():
            if key == target_hp:
                sub_grid[key] = get_space(key)
                if ignore_default:
                    sub_grid[key].remove(get_default_config(key))
            elif best_values_per_hp[key]:
                sub_grid[key] = [best_values_per_hp[key][-1]]
            else:
                sub_grid[key] = [get_default_config(key)]
        return sub_grid
    
    for idx in range(len(config_grid)):
        hp = get_hp_by_rank(idx)
        sub_grid = get_subgrid(hp, ignore_default=idx != 0)

        configs = configs_from_grid(default_config, sub_grid)
        configs_folder = write_configs(configs, gs_root_dir, subspace_prefix=f"hp_{idx}/hp_none")
        slurm_manager = submit_configs(configs_folder, 
                                       sub_grid=sub_grid, 
                                       subspace_prefix=f"hp_{idx}/hp_none", 
                                       max_time=max_time,
                                       gpu_per_job=default_config.train_config.devices)
        if waiting_dimension > 0:
            wait_for_completion(slurm_manager)
            best_config = collect_results(configs_folder, sub_grid=sub_grid, force_reload=False)
            # best_config = collect_results(configs, idx, sub_grid=sub_grid)
            best_values_per_hp[hp].append(best_config[get_hp_name(hp)])

        for i in range(min(idx, repeat_top)):
            sub_grid = get_subgrid(get_hp_by_rank(i), ignore_default=i != idx)

            configs = configs_from_grid(default_config, sub_grid)
            configs_folder = write_configs(configs, gs_root_dir, subspace_prefix=f"hp_{idx}/hp_none")
            slurm_manager = submit_configs(configs_folder, 
                                           sub_grid=sub_grid, 
                                           subspace_prefix=f"hp_{idx}/hp_none", 
                                           max_time=max_time,
                                           gpu_per_job=default_config.train_config.devices)
            if waiting_dimension > 1:
                wait_for_completion(slurm_manager)
                best_config = collect_results(configs_folder, sub_grid=sub_grid, force_reload=False)
                best_values_per_hp[get_hp_by_rank(i)].append(best_config[get_hp_name(get_hp_by_rank(i))])
    return best_values_per_hp

if __name__ == "__main__":
    CLI(sub_grid_search)