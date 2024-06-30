from __future__ import annotations

from pathlib import Path
from typing import Any

from scales.config import PipelineConfig

# from typing import TypeVar
# from scales.config.base_config import BaseConfig

# T = TypeVar("T", bound=BaseConfig)


def collect_configs(configs_folder: Path | None = None, config_pathes: list[Path] | None = None) -> list[Path]:
    config_pathes = [] if config_pathes is None else config_pathes
    configs = []
    if configs_folder:
        for path in configs_folder.iterdir():
            if path.suffix == ".yaml":
                configs.append(path)
        return configs
    return config_pathes


def load_config(config_path: Path) -> PipelineConfig:
    # TODO: Load all types of yaml config files we support here
    return PipelineConfig.from_path(config_path)


def modify_function(config_path: Path, **kwargs: Any) -> PipelineConfig:
    config = load_config(config_path)
    # Modify Config Here
    out_folder = kwargs.pop("output_root_folder", config_path.parent.parent / "output")
    if isinstance(out_folder, str):
        out_folder = Path(out_folder)

    out_path = out_folder / config_path.stem
    config.train_config.load_state_path = out_path  # type: ignore
    # Modify Config End
    return config


def modify_configs(
    configs_folder: Path | None = None,
    config_pathes: list[Path] | None = None,
    output_root_folder: Path | None = None,
    # **kwargs: None,
) -> None:
    config_path_list = collect_configs(configs_folder, config_pathes)
    for path in config_path_list:
        config = modify_function(path, output_root_folder=output_root_folder)
        config.write_yaml(path)


if __name__ == "__main__":
    # Hardcode your modification here
    configs_folder: Path | None = Path(
        "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/pipeline_configs"
    )
    config_pathes: list[Path] | None = None
    output_root_folder: Path | None = Path(
        "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output"
    )
    modify_configs(configs_folder, config_pathes, output_root_folder)
