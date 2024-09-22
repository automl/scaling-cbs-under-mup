from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tbparse import SummaryReader

from scales.config import TrainConfig

MODEL_HPARAMS = ["block_size", "batch_size", "d_model", "n_layer", "n_head", "head_size", "weight_init_type"]


def load_tb(
    output_dir: str | Path = "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output",
    train_config_file_name: str | None = "train_config_post_init", 
    hparams: list[str] = MODEL_HPARAMS,
    force_reload: bool = False
) -> pd.DataFrame:
    output_dir = Path(output_dir)

    if any(["tb_log.csv" in file.name for file in output_dir.iterdir()]) and not force_reload:
    # Return DF if logs are already parsed
        df = pd.read_csv(output_dir / "tb_log.csv", index_col=0)
        return df
    
    log_dir = output_dir / "logs"
    reader = SummaryReader(str(log_dir))
    
    if "value" not in reader.scalars.columns:
        return pd.DataFrame()
    df = pd.pivot_table(reader.scalars, values="value", columns="tag", index="step").ffill()

    if train_config_file_name is None:
        return df

    # Get HPs from written config file
    train_config_path = output_dir / f"{train_config_file_name}.yaml"
    if not train_config_path.exists():
        train_config_path = output_dir / "train_config.yaml"
    if not train_config_path.exists():
        return df
    train_config = TrainConfig.from_path(train_config_path)
    model_config = train_config.model_config

    for hp_name in hparams:
        if hp_name == "batch_size":
            df[hp_name] = train_config.accumulation_iters * train_config.micro_batch_size * train_config.devices
        else:
            df[hp_name] = getattr(model_config, hp_name, getattr(train_config, hp_name, None))
    return df


def read_csv_exp_group(
    exp_group_res_folder: str | Path, 
    train_config_file_name: str | None = "train_config_post_init", 
    hparams: list[str] = MODEL_HPARAMS,
    force_reload: bool = False
) -> pd.DataFrame:
    exp_group_res_folder = Path(exp_group_res_folder)
    exp_paths = []
    exp_names = []
    for root, dirs, files in os.walk(str(exp_group_res_folder)):
        if Path(root).name == "logs" and any("tfevents" in file for file in files) and len(dirs) == 0:
            output_folder = Path(root).parent
            if not "halt" in output_folder.name:
                exp_paths.append(output_folder)
                exp_folder = output_folder.parent if "output" in output_folder.name else output_folder
            # Get relative path of the exp to the root folder as exp_name
                exp_names.append(str(exp_folder.relative_to(exp_group_res_folder)))

    dfs = []
    for path in exp_paths:
        df = load_tb(output_dir=path, train_config_file_name=train_config_file_name, hparams=hparams, force_reload=force_reload)
        dfs.append(df)

    concat_df = pd.concat(dfs, axis=0, keys=exp_names, names=["exp_name", "step"])
    # concat_df.to_csv(str(exp_group_res_folder / "concat_results.csv"))
    return concat_df
