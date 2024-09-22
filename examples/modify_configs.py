from __future__ import annotations

from pathlib import Path
from typing import Any

from scales.config import PipelineConfig


def change_load_state_path(
    config: PipelineConfig, config_path: Path, output_root_folder: Path | str | None
) -> PipelineConfig:
    if output_root_folder is None:
        config.train_config.load_state_path = None  # type: ignore
        return config

    if isinstance(output_root_folder, str):
        output_root_folder = Path(output_root_folder)
    out_path = output_root_folder / config_path.stem
    config.train_config.load_state_path = out_path  # type: ignore
    return config

def add_z_loss_and_wd(config: PipelineConfig,
                      z_loss_eps: float = 1e-4,
                      independent_wd: bool = True) -> PipelineConfig:
    config.train_config.z_loss_eps = z_loss_eps
    config.train_config.independent_wd = independent_wd
    return config

def change_dataset(config: PipelineConfig,
                   hf_dataset_id: str = "",
                   hf_data_subset_name: str = ""):
    config.data_config.hf_dataset_id = hf_dataset_id
    config.data_config.hf_data_subset_name = hf_data_subset_name
    return config


def change_logging(
    config: PipelineConfig,
    global_log_step: int = 1,
    tracked_metrics: dict[str, int] = {},
    log_dir: str | Path | None = None,
    suppress_all_logs: bool = False,
):
    config.train_config.global_log_step = global_log_step
    config.train_config.tracked_metrics = tracked_metrics
    config.train_config.log_dir = log_dir
    config.train_config.suppress_all_logs = suppress_all_logs
    config.train_config.save_state_every = 1000
    return config


def change_lr_to_const(
    config: PipelineConfig,
    cooldown_type: str = "rsqrt",
    cooldown_fraction: float = 0.2,
    scale_lr: bool = False,
    scale_warmup: bool = False,
):
    config.train_config.torch_scheduler = None
    config.train_config.torch_scheduler_args = None
    if scale_lr:
        config.train_config.max_lr = config.train_config.max_lr / 2
    if scale_warmup:
        config.train_config.warmup_fraction = config.train_config.warmup_fraction * 2

    config.train_config.cooldown_type = cooldown_type
    config.train_config.cooldown_fraction = cooldown_fraction
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


def convert_to_new_train_args_hook(config: dict) -> dict:
    if "init_lr" in config:
        config["max_lr"] = config.pop("init_lr")
    if "n_warmup_steps" in config:
        n_warmup_steps = config.pop("n_warmup_steps")
        config["warmup_fraction"] = n_warmup_steps / config["max_train_steps"] if n_warmup_steps is not None else 0
    if "n_cooldown_steps" in config:
        n_cooldown_steps = config.pop("n_cooldown_steps")
        config["cooldown_fraction"] = (
            n_cooldown_steps / config["max_train_steps"] if n_cooldown_steps is not None else 0
        )
    return config


def load_config(config_path: Path) -> PipelineConfig:
    # TODO: Load all types of yaml config files we support here
    return PipelineConfig.from_path(config_path, yaml_hook=convert_to_new_train_args_hook)


def modify_function(config_path: Path, **kwargs: Any) -> PipelineConfig:
    config = load_config(config_path)
    # Modify Config Here
    # output_root_folder = kwargs.pop("output_root_folder", None)
    tracked_metrics = kwargs.pop("tracked_metrics", {})
    config = change_load_state_path(config, config_path, None)
    # config = convert_to_new_lr_param(config)
    config = add_z_loss_and_wd(config)
    # config = change_lr_to_const(config)
    config = change_logging(config, tracked_metrics=tracked_metrics)
    config = change_dataset(config, **kwargs)
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
        "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=2"
    )
    # configs_folder = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/neps_selected/run=1")
    config_pathes: list[Path] | None = None
    # output_root_folder: Path | None = Path(
    #     "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=1"
    # )
    # copy_to_folder: Path | None = "/work/dlclarge1/garibovs-scales_n_arp/configs/SlimPajama-subset_generated=1"
    output_root_folder = None
    # copy_to_folder = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/neps_all/const_lr_no_lr_scale_n_decay_run=1")
    modify_configs(configs_folder, config_pathes, 
                #    output_root_folder=output_root_folder, 
                tracked_metrics={"learning_rate": 1, 
                                 "train_loss": 1, 
                                 "output_logits_max": 10,
                                 "output_logits_mean": 10,
                                 "max_attention_logits_per_layer": 10,
                                 "max_attention_logits_all": 10,
                                 "total_gradient_norm": 20,
                                 "gradient_norm_per_layer": 20,
                                 "validation_loss": 5,
                                 "weight_spectra_max": 5,
                                 "weight_spectra_diff": 5},
                #    copy_to_folder=copy_to_folder,
                   hf_dataset_id="DKYoon/SlimPajama-6B",
                   hf_data_subset_name=""
                   )
