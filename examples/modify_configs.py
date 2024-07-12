from __future__ import annotations

from pathlib import Path
from typing import Any

from scales.config import PipelineConfig

def change_load_state_path(config: PipelineConfig, config_path: Path,
                           output_root_folder: Path | str | None) -> PipelineConfig:
    if output_root_folder is None:
        config.train_config.load_state_path = None  # type: ignore
        return config
    
    if isinstance(output_root_folder, str):
        output_root_folder = Path(output_root_folder)
    out_path = output_root_folder / config_path.stem
    config.train_config.load_state_path = out_path  # type: ignore
    return config

def change_dataset(config: PipelineConfig,
                   hf_dataset_id: str = "",
                   hf_data_subset_name: str = ""):
    config.data_config.hf_dataset_id = hf_dataset_id
    config.data_config.hf_data_subset_name = hf_data_subset_name
    return config


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
    # output_root_folder = kwargs.pop("output_root_folder", None)
    change_load_state_path(config, config_path, **kwargs)
    # change_dataset(config, **kwargs)
    # Modify Config End
    return config


def modify_configs(
    configs_folder: Path | None = None,
    config_pathes: list[Path] | None = None,
    # output_root_folder: Path | None = None,
    copy_to_folder: Path | None = None,
    **kwargs: None,
) -> None:
    if copy_to_folder:
        copy_to_folder = Path(copy_to_folder)
        copy_to_folder.mkdir(exist_ok=True)
    config_path_list = collect_configs(configs_folder, config_pathes)
    for path in config_path_list:
        config = modify_function(path, **kwargs)
        out_path = path
        if copy_to_folder:
            out_path = copy_to_folder / path.name
        config.write_yaml(out_path)


if __name__ == "__main__":
    # Hardcode your modification here
    configs_folder: Path | None = Path(
        "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=1"
    )
    config_pathes: list[Path] | None = None
    output_root_folder: Path | None = Path(
        "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=1"
    )
    copy_to_folder: Path | None = "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=1"
    copy_to_folder = None
    modify_configs(configs_folder, config_pathes, 
                   output_root_folder=output_root_folder, 
                #    copy_to_folder=copy_to_folder,
                #    hf_dataset_id="DKYoon/SlimPajama-6B",
                #    hf_data_subset_name=""
                   )
