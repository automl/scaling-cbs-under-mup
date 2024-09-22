from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonargparse import CLI

from scales.config import PipelineConfig


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


def generate_configs(
    config_root_path: Path, 
    default_config_path: Path, 
    config_grid: dict[str, list[str | int | float]], 
    folder_prefix: str = "generated", 
    config_prefix: str = "pipeline"
) -> None:

    default_pipeline_config = PipelineConfig.from_path(default_config_path)

    config_list = configs_from_grid(default_pipeline_config, config_grid)
    out_path = config_root_path / f"{folder_prefix}=1"
    out_path.mkdir(exist_ok=True, parents=True)

    _ = [
        config.write_yaml(output_dir=out_path / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(config_list)
    ]


def gen_configs_from_list(config_values_list: list[dict[str, Any]], 
                          default_config: PipelineConfig, 
                          config_root_path: Path, 
                          config_prefix: str) -> list[PipelineConfig]:
    
    out_path = config_root_path / f"{config_prefix}=1"
    out_path.mkdir(exist_ok=True)

    for config in config_values_list:
        config_dict = deepcopy(default_config.to_dict(ignore_defaults=True))
        for key_str, value in config.items():
            new_mapping = key_mapping(key_str, value)
            deep_merge(config_dict, new_mapping)
            
        config_values_list.append(PipelineConfig.from_yaml(config_dict))

    _ = [
        config.write_yaml(output_dir=out_path / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(config_values_list)
    ]
    return config_values_list

def config_values_list_scaling() -> list[dict[str, Any]]:
    aspect_ratio = 24
    n_layers_n_heads = [(2, 2), (4, 4), (8, 6), (8, 8), (12, 4), (12, 6), (16, 4), (16, 6), (16, 8), (16, 12), (16, 16)]
    values_list = [{"train_config.model_config.n_layer": n_layers, 
                    "train_config.model_config.n_head": n_heads, 
                    "train_config.model_config.d_model": n_layers*aspect_ratio,
                    "train_config.mup_base_scales.d_model": 32,
                    "train_config.mup_base_scales.n_head": 2,
                    } for n_layers, n_heads in n_layers_n_heads]
    return values_list

def config_values_list_hpo() -> list[dict[str, Any]]:
    
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
    
def config_values_list_lr_wd_grid() -> list[dict[str, Any]]:
    config_grid = {
        "train_config.max_lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "train_config.weight_decay": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-2],
        # "train_config.cooldown_type": ["rsqrt", "linear"],
        # "train_config.cooldown_fraction": [0.01, 0.1, 0.05, 0.2],
        }
    return config_grid
    


if __name__ == "__main__":
    # CLI(generate_configs)
    # generate_configs(Path("some_path"), Path("some_path")
    #                  config_grid={"train_config.model_config.n_embd": [32, 64, 128, 256],
    #                               "train_config.model_config.n_head": [2, 4, 8, 16],
    #                               "train_config.model_config.n_layer": [4, 6, 8, 12]})


    generate_configs(Path("/work/dlclarge1/garibovs-scales_n_arp/configs"), 
                     Path("/work/dlclarge1/garibovs-scales_n_arp/configs/custom=1/def_GS_lr_wd_41M.yaml"),
                     config_values_list_lr_wd_grid(),
                     folder_prefix="grid_search/41M/wd_lr",)
    


    # config_root_path = Path("/work/dlclarge1/garibovs-scales_n_arp/configs")
    # default_config_path = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/custom=1/pipeline_3.yaml")
    # config_prefix = "scale_mup"
    # folder_prefix = "compound_scaling_w_ind_wd_qk_mup"

    # default_config = PipelineConfig.from_path(default_config_path)

    # config_list = []
    # # keys = list(config_grid.keys())
    # # configs_product = itertools.product(*list(config_grid.values()))
    # # values_list = [dict(zip(keys, values)) for values in configs_product]
    # aspect_ratio = 24
    # n_layers_n_heads = [(2, 2), (4,4), (8, 6), (8, 8), (12, 4), (12, 6), (16, 4), (16, 6), (16, 8), (16, 12), (16, 16)]
    # values_list = [{"train_config.model_config.n_layer": n_layers, 
    #                 "train_config.model_config.n_head": n_heads, 
    #                 "train_config.model_config.d_model": n_layers*aspect_ratio,
    #                 "train_config.mup_base_scales.d_model": 32,
    #                 "train_config.mup_base_scales.n_head": 2,
    #                 } for n_layers, n_heads in n_layers_n_heads]
    # for config in values_list:
    #     config_dict = deepcopy(default_config.to_dict(ignore_defaults=True))
    #     for key_str, value in config.items():
    #         new_mapping = key_mapping(key_str, value)
    #         deep_merge(config_dict, new_mapping)

    #     config_list.append(PipelineConfig.from_yaml(config_dict))

    # out_path = config_root_path / f"{folder_prefix}=1"
    # out_path.mkdir(exist_ok=True)

    # _ = [
    #     config.write_yaml(output_dir=out_path / f"{config_prefix}_{i}.yaml")  # type: ignore
    #     for i, config in enumerate(config_list)
    # ]
