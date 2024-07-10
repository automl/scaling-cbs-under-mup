import argparse
from pathlib import Path
from pprint import pprint

import lightning as L
from litgpt import Config

from scales.config.data_config import DataHandler
from scales.config.train_config import PipelineConfig, TrainConfig
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for MuP training")
    parser.add_argument("--width", type=int, default=64, help="width of the GPT model")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / f"output/width{args.width}"

    train_conf = TrainConfig(
        init_lr=0.001,
        micro_batch_size=1,
        block_size=1024,
        weight_decay=0,
        max_val_steps=2,
        n_warmup_steps=None,
        n_main_steps=None,
        n_cooldown_steps=None,
        torch_scheduler="CosineAnnealingLR",
        torch_scheduler_args={"T_max": None, "eta_min": 5e-4},
        model_config=Config(block_size=1024, n_layer=3, n_head=2, vocab_size=50257, bias=True, n_embd=args.width),
        tracked_metrics={
            "train_loss": 1,
            "validation_loss": 5,
            "learning_rate": 1,
            "total_gradient_norm": 10,
            "output_logits_mean": 10,
            "output_logits_max": 10,
            "gradient_norm_per_layer": 20,
            "max_attention_logits_per_layer": 5,
            "max_attention_logits_all": 5,
        },
        load_base_shape_path="width32.bsh",
        max_train_steps=100,
    )

    data_handler = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=True,
        subsample_index=0,
    )

    config = PipelineConfig(data_config=data_handler, train_config=train_conf, eval_config=None)
    config.write_yaml(output_dir)

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
