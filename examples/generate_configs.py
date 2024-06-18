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
    config_root_path: Path, default_config_path: Path, config_grid: dict[str, list[str | int | float]]
) -> None:
    config_prefix = "pipeline"
    folder_prefix = "generated"

    default_pipeline_config = PipelineConfig.from_path(default_config_path)

    config_list = configs_from_grid(default_pipeline_config, config_grid)  # type: ignore
    out_path = config_root_path / f"{folder_prefix}=1"
    out_path.mkdir(exist_ok=True)

    _ = [
        config.write_yaml(output_dir=out_path / f"{config_prefix}_{i}.yaml")  # type: ignore
        for i, config in enumerate(config_list)
    ]


if __name__ == "__main__":
    CLI(generate_configs)
    # generate_configs(Path("some_path"), Path("some_path")
    #                  config_grid={"train_config.model_config.n_embd": [32, 64, 128, 256],
    #                               "train_config.model_config.n_head": [2, 4, 8, 16],
    #                               "train_config.model_config.n_layer": [4, 6, 8, 12]})
