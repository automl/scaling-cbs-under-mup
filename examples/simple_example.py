from pathlib import Path

import lightning as L

from scales.config.data_config import DataHandler, preprocess_wikitext
from scales.config.eval_config import EvalHandler
from scales.lr_utils import LRScheduler
from scales.pretrain_pipeline import main

if __name__ == "__main__":
    fabric = L.Fabric(devices="auto", strategy="auto", accelerator="auto")

    model_dir = Path(__file__).parent / "output"

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-2-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
        subsample_index=0,
    )

    lr_scheduler = LRScheduler(
        init_lr=0.01,
        end_warmup_step=10,
        end_decay_step=40,
        end_cooldown_step=50,
        lr_scheduler="CosineAnnealingLR",
        T_max=30,
        eta_min=5e-4,
    )

    result_dict = main(
        fabric=fabric,
        data=data,
        lr_scheduler=lr_scheduler,
        hparams={"weight_decay": 0.001, "block_size": 512},
        max_train_steps=50,
        max_val_steps=2,
        accumulation_iters=1,
        micro_batch_size=2,
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
