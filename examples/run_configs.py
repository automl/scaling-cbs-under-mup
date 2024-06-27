from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import lightning as L

from scales.config.data_config import DataHandler, preprocess_wikitext
from scales.config.train_config import TrainConfig
from scales.refactored_pretrain import main


def _postprocess_data_handler(
    data_config: DataHandler, root_data_path: str | Path | None = None, seed: int | None = None, block_size: int = 1024
) -> DataHandler:
    data_config.preprocess_fn = preprocess_wikitext
    data_config.root_data_path = Path(root_data_path) if root_data_path is not None else Path("./").absolute() / "data"
    data_config.seed = seed if seed is not None else data_config.seed
    data_config.block_size = block_size
    return data_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("--data_config_path", type=str, required=True, help="Data configuration file.")
    parser.add_argument("--train_config_path", type=str, required=True, help="Training configuration file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    output_dir = Path(__file__).parent / args.output_dir

    # Loading the training configuration
    assert Path(args.train_config_path).exists(), f"Configuration file {args.train_config_path} does not exist!"
    train_config = TrainConfig.from_path(args.train_config_path)
    train_config.log_dir = Path(args.output_dir) / "logs"

    # Load the data configuration
    assert Path(args.data_config_path).exists(), f"Configuration file {args.data_config_path} does not exist!"
    data_config = DataHandler.from_path(args.data_config_path)
    data_config = _postprocess_data_handler(data_config, None, train_config.seed, train_config.block_size)

    pprint(data_config)
    pprint(train_config)

    fabric = L.Fabric(devices="auto", strategy="auto")
    result_dict = main(fabric=fabric, data=data_config, train_args=train_config, out_dir=Path(args.output_dir))

# end of file
