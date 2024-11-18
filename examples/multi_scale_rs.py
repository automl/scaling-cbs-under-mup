from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import time
import itertools
import shutil
from typing import Any
from copy import deepcopy

from jsonargparse import CLI

import yaml
from scales.config import PipelineConfig
from scales.tblog_utils import read_csv_exp_group
from scales.exp_utils.SlurmManager import SlurmManager
import numpy as np
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


def gen_configs_from_list(config_values_list: list[dict[str, Any]], 
                          default_config: PipelineConfig, 
                          config_root_path: Path, 
                          config_prefix: str) -> list[PipelineConfig]:
    
    out_path = config_root_path 
    out_path.mkdir(exist_ok=True)

    pipeline_config_list = []
    for config in config_values_list:
        config_dict = deepcopy(default_config.to_dict(ignore_defaults=True))
        for key_str, value in config.items():
            new_mapping = key_mapping(key_str, value)
            deep_merge(config_dict, new_mapping)
            
        pipeline_config_list.append(PipelineConfig.from_yaml(config_dict))

    _ = [
        config.write_yaml(output_dir=out_path / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(pipeline_config_list)
    ]
    return pipeline_config_list


def load_config_grid(config_grid_path: Path) -> dict[str, dict[str, list[int | float | str] | int | float | str]]:
    with config_grid_path.open("r", encoding="utf-8") as stream:
        config_grid = yaml.safe_load(stream)
    return config_grid

def save_config_grid(config_grid: dict[str, dict[str, list[int | float | str] | int | float | str]], config_grid_path: Path) -> None:
    with config_grid_path.open("w", encoding="utf-8") as stream:
        yaml.dump(config_grid, stream)


def grid_bounds(config_grid: dict[str, dict[str, list[int | float | str] | int | float | str]]) -> dict[str, dict[str, int | float]]:
    bounds = defaultdict(dict)
    for hp, hp_dict in config_grid.items():
        if isinstance(hp_dict["values"][0], (int, float)):
            bounds[hp]["min"] = min(hp_dict["values"])
            bounds[hp]["max"] = max(hp_dict["values"])
        elif isinstance(hp_dict["values"][0], str):
            bounds[hp]["values"] = hp_dict["values"]
        else:
            raise ValueError(f"Unsupported type {type(hp_dict['values'][0])}")
    return bounds


def random_search_single_scale(default_path: Path,
                               config_root_path: Path,
                               n_configs: int,
                               max_time: str = "0-3:00",
                               scale: int = 0) -> None:

    config_grid_path = config_root_path / "config_grid.yaml"
    results_folder = Path(str(config_root_path).replace("/configs/", "/results/"))
    config_grid = load_config_grid(config_grid_path)
    config_grid_path.unlink()
    # bounds = grid_bounds(config_grid)

    config_list = []
    for i in range(n_configs):
        config = {}
        for hp, hp_dict in config_grid.items():
            if hp == "train_config.model_config.rotary_percentage":
                config[hp] = float(np.random.choice(hp_dict["values"]))
            elif isinstance(hp_dict["values"][0], (float)):
                if hp_dict.get("log_scale", False):
                    value = 10 ** np.random.uniform(np.log10(min(hp_dict["values"])), np.log10(max(hp_dict["values"])))
                else:
                    value = np.random.uniform(min(hp_dict["values"]), max(hp_dict["values"]))
                config[hp] = float(value)
            elif isinstance(hp_dict["values"][0], bool):
                config[hp] = bool(np.random.choice([True, False]))
            # elif isinstance(hp_dict["values"][0], int):
            #     if hp_dict.get("log_scale", False):
            #         value = np.exp(np.random.uniform(np.log10(min(hp_dict["values"])), np.log10(max(hp_dict["values"]))))
            #     else:
            #         value = np.random.randint(min(hp_dict["values"]), max(hp_dict["values"]))
            #     config[hp] = int()
            elif isinstance(hp_dict["values"][0], str):
                config[hp] = str(np.random.choice(hp_dict["values"]))
            
            else:
                raise ValueError(f"Unsupported type {type(hp_dict['values'][0])}")
        config_list.append(config)

    default_pipeline_config = PipelineConfig.from_path(default_path)
    
    pipeline_configs = gen_configs_from_list(config_list,
                                    default_pipeline_config,
                                    config_root_path,
                                    "pipeline")
    
    slurm_manager = SlurmManager(experiment_name=f"RS_{scale}",
                                 )
    slurm_manager.create_slurm_script_folder(partition="bosch_gpu-rtx2080",
                                             max_time=max_time,
                                             gpu_per_job=default_pipeline_config.train_config.devices,
                                             root_output_path=str(results_folder),
                                             root_configs_path=str(config_root_path))
    if not slurm_manager.check_run_completed():
        slurm_manager.submit()
    

def random_search(default_paths: list[Path],
                  config_root: Path,
                  n_configs: int,
                  scaling: int = 5,
                  max_times: list[str] | None = None) -> None:
    
    # config_grid = load_config_grid(config_grid_path)
    # bounds = grid_bounds(config_grid)
    if max_times is None:
        max_times = ["0-06:00", "0-12:00", "0-24:00", "2-00:00"]

    for idx, default_path in enumerate(default_paths):
        print(f" Scale: {idx}, default_path: {default_path}")
        scale_path = config_root / f"scale={idx}"
        scale_path.mkdir(exist_ok=True)
        shutil.copy(str(config_root / "config_grid.yaml"), str(scale_path / "config_grid.yaml"))
        scaling_power = len(default_paths) - idx - 1
        random_search_single_scale(default_path,
                                   scale_path,
                                   n_configs * (scaling ** scaling_power),
                                   max_time=max_times[idx],
                                   scale=idx)


if __name__ == "__main__":
    default_paths = [
        Path("/work/dlclarge1/garibovs-scales_n_arp/configs/defaults/s_loss_exp=1/chinchilla_pipeline_5M.yaml"),
        Path("/work/dlclarge1/garibovs-scales_n_arp/configs/defaults/s_loss_exp=1/chinchilla_pipeline_25M.yaml"),
        Path("/work/dlclarge1/garibovs-scales_n_arp/configs/defaults/s_loss_exp=1/chinchilla_pipeline_125M.yaml"),
        # Path("/work/dlclarge1/garibovs-scales_n_arp/configs/defaults/scaling=5/chinchilla_pipeline_625M.yaml"),
    ]
    random_search(default_paths, Path("/work/dlclarge1/garibovs-scales_n_arp/configs/rs/s_loss_exp=1"), 2, 2, 
                  ["0-06:00", "0-12:00", "0-24:00", 
                #    "2-00:00"
                   ])
