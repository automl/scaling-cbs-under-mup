from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import time
import itertools
from typing import Any, Callable
from copy import deepcopy

from jsonargparse import CLI

import yaml
from scales.config import PipelineConfig
from scales.tblog_utils import read_csv_exp_group
from scales.SlurmManager import SlurmManager
import pandas as pd


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
from scales.SlurmManager import SlurmManager
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

class PipelineFactory:
    def __init__(self):
        pass
    
    def write_configs(self, configs: list[PipelineConfig], output_dir: Path) -> None:
        """Write the configs to the output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, config in enumerate(configs):
            config_path = output_dir / f"pipeline_{i}.yaml"
            config.write_yaml(config_path, ignore_defaults=True)

    def read_configs(self, config_dir: Path) -> list[PipelineConfig]:
        """Read the configs from the config_dir."""
        configs = []
        for config_path in config_dir.glob("pipeline_*.yaml"):
            configs.append(PipelineConfig.from_path(config_path))
        return configs

    def check_all_configs_exist(self, configs: list[PipelineConfig], output_dir: Path) -> bool:
        """Check if all the configs exist in the output_dir."""
        raise NotImplementedError("Implement BaseConfig.__eq__ first")
        existing_configs = self.read_configs(output_dir)
        # return all()
        return all(config in existing_configs for config in configs)
    
    def gen_configs_from_list(self,
                              config_values_list: list[dict[str, Any]], 
                              default_config: PipelineConfig) -> list[PipelineConfig]:
        """Generate a list of PipelineConfigs from a list of dictionaries."""

        pipeline_config_list = []
        for config in config_values_list:
            config_dict = deepcopy(default_config.to_dict(ignore_defaults=True))
            for key_str, value in config.items():
                new_mapping = key_mapping(key_str, value)
                deep_merge(config_dict, new_mapping)
                
            pipeline_config_list.append(PipelineConfig.from_yaml(config_dict))

        return pipeline_config_list

    def output_configs(self,
                    default_config: PipelineConfig, 
                    config_grid: dict[str, list[str | int | float]],
                    output_dir: Path,
                    sampler: Callable[..., list[dict[str, Any]]]) -> None:
        """Output the configs to the output_dir."""
        
        configs_list = sampler(config_grid)

        pipeline_config_list = self.gen_configs_from_list(configs_list, default_config)

        self.write_configs(pipeline_config_list, output_dir)

        



