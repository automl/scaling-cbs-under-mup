import argparse
import os
import re
from pathlib import Path
from pprint import pprint

import lightning as L
import pandas as pd
from litgpt import Config

from scales.config.data_config import DataHandler
from scales.config.train_config import PipelineConfig, TrainConfig
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main


def find_best_lr_or_default(base_directory, default_lr, strict_lr, width_base) -> float:
    best_lr = None
    best_val_loss = float("inf")

    if strict_lr:
        return float(default_lr)

    try:
        for folder in os.listdir(base_directory):
            if folder.startswith(f"tune_lr_w{width_base}_bs"):
                folder_path = os.path.join(base_directory, folder)

                # Check if the folder contains the summary CSV file
                csv_path = os.path.join(folder_path, "summary_csv", "config_data.csv")
                if os.path.isfile(csv_path):
                    df = pd.read_csv(csv_path)
                    # Ensure the required columns exist
                    if hasattr(df, "result.val_loss") and hasattr(df, "result.lr"):
                        min_row = df.loc[df["result.val_loss"].idxmin()]
                        val_loss = min_row["result.val_loss"]
                        lr = min_row["result.lr"]

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_lr = lr

        if best_lr is None:
            raise ValueError("No valid learning rate found.")

        return float(best_lr)

    except Exception as e:
        print(f"Error encountered: {e}. Using default value.")
        return float(default_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for MuP training")
    parser.add_argument("--width_base", type=int, default=32, help="width of the base GPT model")
    parser.add_argument("--width_target", type=int, default=288, help="width of the target GPT model")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.013894954943731374,
        help="The scaled LR",
    )
    parser.add_argument("--strict_default_lr", action="store_true", help="Use default learning rate value")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="The micro batch size")
    parser.add_argument("--block_size", type=int, default=1024, help="The block size of the model")
    parser.add_argument("--total_validation_tokens", type=int, default=2500000, help="Number of validation tokens")
    parser.add_argument("--n_gpus", type=int, default=1, help="The number of gpus used")
    parser.add_argument("--array_index", type=int, help="Index from SLURM array job")
    parser.add_argument("--tokens_per_param", type=int, default=20, help="Number of tokens to be trained per param")
    parser.add_argument("--seed", type=int, default=444, help="SEED to run at")
    args = parser.parse_args()

    effective_batch_size_array = [16, 32, 64, 128, 256, 512, 1024, 2048]

    total_validation_steps = int(args.total_validation_tokens / (args.micro_batch_size * args.n_gpus * args.block_size))

    base_dir = f"results/"
    learning_rate = find_best_lr_or_default(base_dir, args.learning_rate, args.strict_default_lr, args.width_base)

    output_dir = (
        Path(__file__).parent
        / f"output/final_results/seed{args.seed}/mup_wb{args.width_base}_wt{args.width_target}_tpp{args.tokens_per_param}/bs{effective_batch_size_array[args.array_index]}"
    )

    accumulation_iters = int(effective_batch_size_array[args.array_index] / (args.n_gpus * args.micro_batch_size))

    current_ebs = accumulation_iters * args.n_gpus * args.micro_batch_size

    if effective_batch_size_array[args.array_index] != current_ebs:
        raise ValueError

    train_conf = TrainConfig(
        max_lr=float(learning_rate),
        micro_batch_size=int(args.micro_batch_size),
        seed=args.seed,
        weight_init_type="GPT-NeoX",
        block_size=args.block_size,
        weight_decay=0,
        max_val_steps=total_validation_steps,
        validate_every=20,
        accumulation_iters=accumulation_iters,
        model_config=Config(
            block_size=args.block_size,
            n_layer=8,
            n_head=4,
            vocab_size=50257,
            bias=True,
            n_embd=args.width_target,
        ),
        tracked_metrics={
            "train_loss": 1,
            "validation_loss": 20,
            "learning_rate": 1,
            "total_gradient_norm": 5,
            "output_logits_mean": 10,
            "output_logits_max": 10,
            "gradient_norm_per_layer": 20,
            "max_attention_logits_per_layer": 5,
            "max_attention_logits_all": 5,
            "optimizer_stats": 1,
            "tokens_per_step": 1,
            "activations_train": 1,
            "layerwise_features_rms_val": 1,
            "layerwise_features_l1_mean_val": 1,
        },
        mup_base_scales=args.width_base,
        load_state_path=output_dir,
        recovery_state=True,
        save_init_state=False,
        tokens_per_param=args.tokens_per_param,
    )

    data_handler = DataHandler(
        hf_dataset_id="DKYoon/SlimPajama-6B",
        hf_data_subset_name="",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=True,
        subsample_index=0,
    )

    config = PipelineConfig(data_config=data_handler, train_config=train_conf, eval_config=None)
    fabric = L.Fabric(devices="auto", strategy="auto")

    data_handler = config.data_config

    pprint(config.train_config)

    if isinstance(data_handler, DataHandler):
        result_dict = main(
            fabric=fabric,
            data=data_handler,
            train_args=config.train_config,  # type: ignore
            out_dir=output_dir,
        )
    else:
        raise ValueError("Error loading the data")
