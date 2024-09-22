from __future__ import annotations
import logging
import torch
import lightning as L
from pathlib import Path

from scales.refactored_pretrain import main
from scales.config import TrainConfig, DataHandler

DEFAULT_TRAIN_CONFIG_FILE = "train_config.yaml"
DEFAULT_DATA_CONFIG_FILE = "data_config.yaml"


def dynamic_micro_batch(output_dir: Path | str,):
    
    output_dir = Path(output_dir)
    train_config = TrainConfig.from_path(output_dir / DEFAULT_TRAIN_CONFIG_FILE)
    data = DataHandler.from_path(output_dir / DEFAULT_DATA_CONFIG_FILE)
    fabric = L.Fabric(strategy="auto", devices="auto")
    fits_memory = False
    while not fits_memory:
        try:
            train_config.write_yaml(output_dir / DEFAULT_TRAIN_CONFIG_FILE, ignore_defaults=True)

            result_dict = main(fabric=fabric, data=data, train_args=train_config, out_dir=output_dir)
            fits_memory = True
        except torch.cuda.OutOfMemoryError as e:
            if train_config.micro_batch_size <= 1:
                raise MemoryError("The model does not fit memory even with a single data sequence") from e
            train_config.micro_batch_size = int(train_config.micro_batch_size / 2)
            train_config.accumulation_iters = int(train_config.accumulation_iters * 2)
            logging.info(f"training ran out of memory trying micro_batch_size: {train_config.micro_batch_size}")
            fits_memory = False

    return result_dict