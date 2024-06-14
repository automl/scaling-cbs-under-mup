from __future__ import annotations

import re
from pathlib import Path

import torch
import yaml
from jsonargparse import CLI
from lightning import Fabric

from scales.config import PipelineConfig
from scales.refactored_pretrain import main


def run(
    config_path: str | Path,
    output_root_dir: str | Path,
    data_root_path: str | Path | None = None,
    access_internet: bool = False,
) -> None:
    """Run the configuration, and save results.

    :param config_path: Path to the Pipeline config yaml file
    :param data_root_path: Training data tree root path
    :param output_root_dir: output tree root path
    :param access_internet: Set True to download the data if it doesn't exist, otherwise throw an error
    :return:

    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
        assert config_path.exists(), f"Configuration at {config_path} does not exist"
    if data_root_path is not None and isinstance(data_root_path, str):
        data_root_path = Path(data_root_path)
        assert data_root_path.exists(), f"The root data folder {data_root_path} does not exist"
    if isinstance(output_root_dir, str):
        output_root_dir = Path(output_root_dir)
        assert output_root_dir.exists(), f"The root_output_dir {output_root_dir} does not exist"
        assert output_root_dir.is_dir(), "output_root_dir must be a directory"

    output_path = output_root_dir / config_path.stem
    # TODO: what to do when there is already a run in output_path ?
    output_path.mkdir(exist_ok=True)

    pipe_config = PipelineConfig.from_path(config_path)
    pipe_updated = False

    if data_root_path is not None and pipe_config.data_config.root_data_path != data_root_path:
        # Update the root path for the data (since it's cluster specific)
        pipe_config.data_config.root_data_path = data_root_path
        pipe_updated = True

    fabric = Fabric(devices="auto", strategy="auto")

    data_handler = pipe_config.data_config
    train_config = pipe_config.train_config
    # eval_handler = pipe_config.eval_config
    try:
        result = main(
            fabric=fabric,
            train_args=train_config,
            data=data_handler,
            out_dir=output_path,
            access_internet=access_internet,
        )
        result_path = output_path / "result.yaml"
        with result_path.open(mode="w", encoding="utf-8") as file:
            yaml.dump(result, file)
    except torch.cuda.OutOfMemoryError:
        # in case of a memory error bump the config to the next config group
        # Expected config parent folder name: "some_text=some_number"
        matching = re.search(r"([^=]+)=([\d]+)", str(config_path.parent.name))
        if matching is not None:
            folder_prefix, folder_order = matching.group(1), int(matching.group(2))
            folder_order += 1

            # Move the config to the new folder with name "some_text=some_number + 1"
            new_config_folder = config_path.parent.parent / f"{folder_prefix}={folder_order}"
            new_config_folder.mkdir(exist_ok=True)
            new_config_path = new_config_folder / config_path.name
            config_path = config_path.rename(new_config_path)

    if pipe_updated:
        # Write the updated yaml back
        pipe_config.write_yaml(output_dir=config_path)


if __name__ == "__main__":
    CLI(run)
