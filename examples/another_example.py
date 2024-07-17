from pathlib import Path
from pprint import pprint

import lightning as L
from litgpt.config import Config

from scales.config.data_config import DataHandler
from scales.config.train_config import PipelineConfig, TrainConfig
from scales.config.utils import preprocess_wikitext
from scales.refactored_pretrain import main

if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"

    if (output_dir / "PipelineConfig.yaml").exists():
        config = PipelineConfig.from_path(output_dir / "PipelineConfig.yaml")
    else:
        # Define your own PipelineConfig
        train_conf = TrainConfig(
            max_lr=0.01,
            micro_batch_size=1,
            block_size=128,
            weight_decay=0.001,
            max_val_steps=2,
            warmup_fraction=0.2,
            cooldown_fraction=0.2,
            torch_scheduler="CosineAnnealingLR",
            torch_scheduler_args={"eta_min": 5e-4},
            model_config=Config(block_size=128, vocab_size=50257, n_embd=32, n_layer=3, n_head=2),
            # tokens_per_param=20,
            max_train_steps=100,
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

        eval_config = None
        # Optionally define Eval config
        from scales.config.eval_config import EvalHandler

        eval_config = EvalHandler(
            model_dir=output_dir,
            tokenizer_dir=data_handler.tokenizer_root_path / data_handler.tokenizer_repo_id,
            lm_eval_tasks="mmlu_professional_law",
        )

        # Optionally save configs separately
        eval_config.write_yaml(output_dir=output_dir / "can_name_them_too.yaml")
        # data_handler.write_yaml(data_handler.binary_path)
        # Or use the Class name as default
        train_conf.write_yaml(output_dir=output_dir)

        config = PipelineConfig(data_config=data_handler, train_config=train_conf, eval_config=eval_config)
        config.write_yaml(output_dir)

    fabric = L.Fabric(devices="auto", strategy="auto")

    data_handler = config.data_config  # type: ignore

    pprint(config.train_config)

    result_dict = main(
        fabric=fabric,
        data=data_handler,
        train_args=config.train_config,  # type: ignore
        out_dir=output_dir,
    )

    if config.eval_config:
        config.eval_config.evaluate()
