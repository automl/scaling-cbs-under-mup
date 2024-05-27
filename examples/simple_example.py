from pathlib import Path

import lightning as L

from scales.args import LoggingArgs
from scales.data_utils import DataHandler, preprocess_wikitext
from scales.eval_pipeline import EvalHandler
from scales.lr_utils import ExponetialWarmupSchedulerLR
from scales.pretrain_pipeline import main

if __name__ == "__main__":
    fabric = L.Fabric(devices="auto", strategy="auto")

    model_dir = Path(__file__).parent / "output"

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
        subsample_index=0,
    )

    lr_details = ExponetialWarmupSchedulerLR(
        init_lr=0.01, decay_rate=None, min_lr=5e-6, max_warmup_steps=150, start_decay_at_step=1500, max_decay_steps=500
    )

    result_dict = main(
        fabric=fabric,
        data=data,
        lr_details=lr_details,
        logging=LoggingArgs(
            train_loss=True,
            validation_loss=True,
            learning_rate=True,
            total_gradient_norm=True,
            output_logits_mean=True,
            log_step=4,
        ),
        hparams={"weight_decay": 0.001, "batch_size": 4, "block_size": 1024},
        max_train_steps=2000,
        max_val_steps=2,
        max_norm=1.0,
        out_dir=model_dir,
        model_config_file=Path(__file__).parent / "model.yaml",
    )

    fabric.print(f"Final Validation loss: {result_dict['val_loss']}")

    EvalHandler(
        model_dir=model_dir,
        tokenizer_dir=data.tokenizer_root_path / data.tokenizer_repo_id,
        lm_eval_tasks="mmlu_professional_law",
    ).evaluate()
