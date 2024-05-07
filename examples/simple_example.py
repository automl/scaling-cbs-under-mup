from pathlib import Path

import lightning as L

from scales.data_utils import DataHandler, preprocess_wikitext
from scales.pretrain_pipeline import main

if __name__ == "__main__":
    fabric = L.Fabric(devices="auto", strategy="auto")

    data = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-2-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_splits=True,
    )

    result_dict = main(
        fabric=fabric,
        data=data,
        hparams={"lr": 0.002, "weight_decay": 0.001, "batch_size": 2, "block_size": 2048},
        max_train_steps=300,
        max_val_steps=2,
        model_config_file=Path(__file__).parent / "model.yaml",
    )

    fabric.print(f"Final Validation loss: {result_dict['val_loss']}")
