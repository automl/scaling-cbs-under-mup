from __future__ import annotations

from pathlib import Path
from jsonargparse import CLI
from scales.config import DataHandler
from scales.config.data_config import preprocess_wikitext

def download_dataset(hf_dataset_id: str, 
                     hf_data_subset_name: str, 
                     root_data_path: str | Path,
                     force_splits: bool = True,
                     use_subsamples: bool = False) -> None:
    """Download the dataset from Hugging Face and preprocess it."""
    if use_subsamples:
        subsample_index = 1
    else:
        subsample_index = 0
    
    if isinstance(root_data_path, str):
        root_data_path = Path(root_data_path)

    data_handler = DataHandler(
        hf_dataset_id=hf_dataset_id,
        hf_data_subset_name=hf_data_subset_name,
        tokenizer_repo_id="openai-community/gpt2",
        root_data_path=root_data_path,
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=force_splits,
        subsample_index=subsample_index,
    )
    data_handler.load_data_loaders(access_internet=True)
    return None

if __name__ == "__main__":
    CLI(download_dataset)