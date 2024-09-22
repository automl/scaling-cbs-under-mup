from __future__ import annotations

from pathlib import Path

from scales.config import DataHandler, PipelineConfig, TrainConfig


def convert_to_new_train_args_hook(config: dict) -> dict:
    config.pop("n_main_steps", None)
    if "init_lr" in config:
        config["max_lr"] = config.pop("init_lr")
    if "n_warmup_steps" in config:
        n_warmup_steps = config.pop("n_warmup_steps")
        config["warmup_fraction"] = n_warmup_steps / config["max_train_steps"] if n_warmup_steps is not None else 0
    if "n_cooldown_steps" in config:
        n_cooldown_steps = config.pop("n_cooldown_steps")
        config["cooldown_fraction"] = n_cooldown_steps / config["max_train_steps"] if n_cooldown_steps is not None else 0
    return config

def read_train_configs(neps_root_dir: Path, collection_path: Path, ids: list[int | str], dataset_path: Path) -> None:
    """Read the train configs from the NEPS run and save them in the collection_path folder.

    Args:
        neps_root_dir (Path): The root directory of the NEPS run
        collection_path (Path): The path to the folder where the configs should be saved
        ids (list[int | str]): The ids of the configs to be saved
        dataset_path (Path): The path to the dataset config. Not the data root directory, but the dataset itself

    """

    collection_path.mkdir(parents=True, exist_ok=True)

    for id_ in ids:
        config_dir = neps_root_dir / "results" / f"config_{id_}"

        config_file = (
            config_dir / "train_config.yaml"
            if (config_dir / "train_config.yaml").exists()
            else config_dir / "output" / "train_config.yaml"
        )
        if not config_file.exists():
            continue

        train_config = TrainConfig.from_path(config_file, yaml_hook=convert_to_new_train_args_hook)
        # load dataset config
        data_config = DataHandler.from_path(dataset_path)
        # train_config = pipeline_config.train_config

        # Set the save_state_path to None to avoid loading the model
        train_config.save_state_path = None

        pipeline_config = PipelineConfig(train_config=train_config, data_config=data_config)

        config_path = collection_path / f"pipeline_{id_}.yaml"

        pipeline_config.write_yaml(config_path, ignore_defaults=True)


if __name__ == "__main__":
    neps_root_dir = Path("/work/dlclarge1/garibovs-scales_n_arp/results/neps/run_1")
    collection_path = Path("/work/dlclarge1/garibovs-scales_n_arp/configs/neps_all/run=1")
    dataset_path = Path("/work/dlclarge1/garibovs-scales_n_arp/scaling_all_the_way/data/binaries/DKYoon/SlimPajama-6B")
    # Selected config ids
    ids = [12, 13, 49, 57, 50, 25, 2, 46, 23, 37, 30, 54, 28, 9, 14, 29, 17, 32, 52, 53, 20, 56]
    ids = list(range(0, 60))

    read_train_configs(neps_root_dir, collection_path, ids, dataset_path)
