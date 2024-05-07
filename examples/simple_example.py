from pathlib import Path

import lightning as L

from scales.data_utils import DataHandler, preprocess_wikitext
from scales.eval_pipeline import EvalHandler
from scales.pretrain_pipeline import main

if __name__ == "__main__":
    fabric = L.Fabric(devices="auto", strategy="auto")

    model_dir = Path(__file__).parent.parent / "output"

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
    )

    result_dict = main(
        fabric=fabric,
        data=data,
        hparams={"lr": 0.002, "weight_decay": 0.001, "batch_size": 4, "block_size": 1028},
        max_train_steps=2000,
        max_val_steps=2,
        out_dir=model_dir,
        model_config_file=Path(__file__).parent / "model.yaml",
    )

    fabric.print(f"Final Validation loss: {result_dict['val_loss']}")

    EvalHandler(
        model_dir=model_dir,
        tokenizer_dir=data.tokenizer_root_path / data.tokenizer_repo_id,
        lm_eval_tasks="mmlu_professional_law",
    ).evaluate()
