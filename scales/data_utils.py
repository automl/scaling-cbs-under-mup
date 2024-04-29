from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Dict, List

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from litdata.processing.functions import optimize
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset
from litgpt.tokenizer import Tokenizer
from torch import Tensor


def download_tokenizer(repo_id: str, root_dir: str | Path, overwrite: bool = False) -> None:
    """Download the trained tokenizer from the selected HF repo. The tokwenizer will be saved under /root_dir/repo_id.

    Note: To use HF token for authentication set the environment variable HF_TOKEN

    Args:
    ----
    repo_id (str): HuggingFace repository id for the tokenizer to be downloaded from
    root_dir (Path, str): path to save the tokenizer under

    """
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    if (root_dir / repo_id).exists() and not overwrite:
        # Tokenizer already exists
        return
    from litgpt.scripts.download import download_from_hub

    download_from_hub(repo_id=repo_id, tokenizer_only=True, checkpoint_dir=root_dir, convert_checkpoint=False)


def preprocess_wikitext(row: Dict[str, str]) -> str:
    """Wikitext specific preprocessing, removes new_line at the end of each file."""
    return re.sub(r"\n$", "", row["text"])


def tokenize_wikitext(index: int, dataset: Dataset, tokenizer: Tokenizer, prep_fn: Callable) -> Tensor:
    """Tokenize each row in the dataset separately."""
    # only yield for now due to a bug on litdata
    # https://github.com/Lightning-AI/litdata/issues/70
    yield tokenizer.encode(prep_fn(dataset[index]), eos=True)


# def tokenize_text() -> None:
#
#     train_dataset = load_dataset("wikitext", "wikitext-2-v1", split="train").filter(lambda row: len(row["text"]) > 1)
#     print(len(train_dataset))
#
#     repo_id = "meta-llama/Llama-2-7b-hf"
#     repo_id = "openai-community/gpt2"
#     path = Path("/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scales-n-arpeggios/data")
#     out = "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scales-n-arpeggios/out_temp"
#     download_tokenizer(repo_id=repo_id, root_dir=path)
#     tokenizer = Tokenizer(path / repo_id)
#
#     token_fn = partial(tokenize_wikitext, dataset=train_dataset, tokenizer=tokenizer, prep_fn=preprocess_wikitext)
#
#     print(token_fn(0))
#
#     optimize(fn=token_fn, inputs=list(range(len(train_dataset))), output_dir=out, chunk_bytes="64MB")


@dataclass
class DataHandler:
    # TODO: Write a wrapper to read configs from a yaml file and do everything
    hf_dataset_id: str
    """Dataset identifier for the HuggingFace Datasets."""
    hf_data_subset_name: str
    """Dataset Subset name for the datasets with multiple subsets."""
    preprocess_fn: Callable
    """`tokenizer_fn` usually requires a preprocess function which takes in in a single argument which is a row of the
    dataset being processed and outputs the processed text file."""
    tokenizer_repo_id: str
    """HuggingFace repository ID for the model which we will use the tokenizer of."""
    hf_data_files: str | None = None
    """HuggingFace dataset data_files."""
    filter_function: Callable = field(default=lambda row: len(row["text"]) > 1)
    """Filter function to be called on each dataset object immediately after loading."""
    tokenizer_fn: Callable = tokenize_wikitext
    """Tokenizer function to be used for the `optimize` function of `litdata.processing.functions` takes in at least a
    single argument which is the index of the dataset and returns tokenized text for that index."""
    root_data_path: Path = Path(__file__).parent.parent / "data"
    """Root folder which holds tokenizers, binaries, cache folders."""
    splits: List[str] = field(default_factory=lambda: ["train", "validation", "test"])
    """Data splits to create or load from hf hub.

    If any split other than the 'train' split doesn't exist,
    the loader will try to load all the splits and split them according to the `default_split_ratio`.
    If 'train' split is specified but not found then an exception will occur

    """
    default_split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    """Split ratios for data splits when either a split specified is not found on the hub or `force_splits` flag is
    set."""
    force_splits: bool = False
    """Forces the loader to load all the splits existing on the hub for the dataset and split them according to the
    `default_split_ratio`"""
    force_overwrite: bool = False
    """Forces to process the dataset again if set."""
    tokenizer_fn_kwargs: Dict[Any, Any] = field(default_factory=dict)
    """`tokenizer_fn` can take multiple arguments if their defaults are specified here."""
    seed: int = 42
    """Seed for splitting datasets and shuffling."""

    nlp_dataset: bool = True
    """Use optimized data reading for the `StreamingDataset` if the dataset is an nlp dataset."""
    batch_size: int = 64
    """Batch size for `StreamingDataLoader` (same for all splits)"""
    block_size: int = 2048
    """Block size in each batch."""
    num_workers: int = 1
    """Number of workers for `StreamingDataLoader`"""

    # TODO: organize filter calls to be consistent

    def __post_init__(self) -> None:
        self.tokenizer_root_path = self.root_data_path / "tokenizers"
        """Root folder where to store all the downloaded tokenizers."""
        self.bin_data_path = self.root_data_path / "binaries"
        """Root folder where to store all the tokenized datasets."""
        self.cache_dir = self.root_data_path / "cache"
        """Cache directory for huggingface datasets."""
        self.binary_path = self.bin_data_path / self.hf_dataset_id / self.hf_data_subset_name
        self.datasets: Dict[str, StreamingDataset] = {}
        self.data_loaders: Dict[str, StreamingDataLoader] = {}

    def __convert_to_binary(self) -> None:
        if (
            not self.force_overwrite
            and all((self.binary_path / split).exists() for split in self.splits)
            and self.serialized() == self.load_yaml()
        ):
            # Return if all folders for splits exists and serialized version of the config matches the yaml file
            warnings.warn(
                f"Dataset {self.hf_dataset_id} already exists at {self.binary_path}.\n"
                f"Quitting...\n"
                f"If you would like to prepare the dataset again set the `force_overwrite` flag."
            )
            return

        dataset_splits = self.__get_data_splits()

        download_tokenizer(repo_id=self.tokenizer_repo_id, root_dir=self.tokenizer_root_path)
        tokenizer = Tokenizer(self.tokenizer_root_path / self.tokenizer_repo_id)

        for split, dataset in dataset_splits.items():
            split_output_path = self.binary_path / split
            split_output_path.mkdir(parents=True, exist_ok=True)
            token_fn = partial(
                self.tokenizer_fn,
                dataset=dataset,
                tokenizer=tokenizer,
                prep_fn=self.preprocess_fn,
                **self.tokenizer_fn_kwargs,
            )
            optimize(
                fn=token_fn, inputs=list(range(len(dataset))), output_dir=str(split_output_path), chunk_bytes="64MB"
            )
        self.write_yaml()

    def __get_data_splits(self) -> Dict[str, Dataset]:
        dataset_splits: Dict[str, Any] = {}
        missing_splits: bool = False
        if not self.force_splits:
            for split in self.splits:
                try:
                    dataset: Dataset = load_dataset(
                        path=self.hf_dataset_id,
                        name=self.hf_data_subset_name,
                        split=split,
                        data_files=self.hf_data_files,
                        streaming=False,
                        cache_dir=str(self.cache_dir),
                    ).filter(self.filter_function)
                    dataset_splits[split] = dataset
                except ValueError as e:
                    # Don't load if the train split doesn't exist
                    if split == "train":
                        raise ValueError(
                            f"There was an error during loading the train split of the dataset: "
                            f"{self.hf_dataset_id}: {self.hf_data_subset_name}"
                        ) from e
                    warnings.warn(
                        f"The split: {split} of the dataset: {self.hf_dataset_id}: {self.hf_data_subset_name} "
                        f"couldn't be loaded."
                    )
                    missing_splits = True
                    dataset_splits = {}
                    break

        if missing_splits or self.force_splits:
            dataset_dict: DatasetDict = load_dataset(
                path=self.hf_dataset_id,
                name=self.hf_data_subset_name,
                data_files=self.hf_data_files,
                cache_dir=str(self.cache_dir),
            )
            combined_dataset = concatenate_datasets(
                [dataset_dict[split].filter(self.filter_function) for split in list(dataset_dict.keys())]
            )
            if len(self.splits) < 2:
                dataset_splits[self.splits[0]] = combined_dataset
            else:
                assert sum(self.default_split_ratio) == 1, "Split ratios must sum up to 1"
                # Split combined data into len(self.default_split_ratio) splits
                split_ratio_pairs = sorted(zip(self.default_split_ratio, self.splits), reverse=True)
                remaining_dataset = combined_dataset
                remainder = 1.0
                for ratio, split in split_ratio_pairs[:-1]:
                    dataset_dict_: DatasetDict = remaining_dataset.train_test_split(
                        test_size=1 - (ratio / remainder), seed=self.seed
                    )
                    target_dataset, remaining_dataset = dataset_dict_["train"], dataset_dict_["test"]
                    dataset_splits[split] = target_dataset
                    remainder = 1 - (ratio / remainder)
                dataset_splits[split_ratio_pairs[-1][-1]] = remaining_dataset

        return dataset_splits

    def __load_datasets(self) -> None:
        """Loads the `StreamingDatasets` into `self.datasets` dict, keys are `splits`"""
        # TODO: Add option to merge datasets with `CombinedStreamingDataset`
        item_loader = None
        if self.nlp_dataset:
            from litdata.streaming.item_loader import TokensLoader

            item_loader = TokensLoader(block_size=self.block_size + 1)

        for split in self.splits:
            self.datasets[split] = StreamingDataset(
                input_dir=str(self.binary_path / split), item_loader=item_loader, shuffle=True, seed=self.seed
            )

    def load_data_loaders(self) -> None:
        """Loads the `StreamingDataLoaders` into `self.data_loaders` dict, keys are `splits`"""
        self.__convert_to_binary()
        self.__load_datasets()
        for split in self.splits:
            self.data_loaders[split] = StreamingDataLoader(
                dataset=self.datasets[split], batch_size=self.batch_size, num_workers=self.num_workers
            )

    def serialized(self) -> Dict[str, Any]:
        # TODO: make the class reconstructable from the yaml file
        def serialize(value: Any) -> Any:
            if isinstance(value, partial):
                return {"function": f"{value.func.__module__}.{value.func.__name__}", "kwargs": value.keywords}
            if isinstance(value, FunctionType):
                return f"{value.__module__}.{value.__name__}"
            if isinstance(value, Path):
                return str(value)
            return value

        return {key: serialize(value) for key, value in asdict(self).items()}

    def write_yaml(self) -> None:
        ser_dict = self.serialized()
        with (self.binary_path / "dataset.yaml").open("w", encoding="utf-8") as yaml_file:
            yaml.dump(ser_dict, yaml_file)

    def load_yaml(self) -> Dict[str, Any]:
        with (self.binary_path / "dataset.yaml").open(encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)


if __name__ == "__main__":
    data_handler = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-2-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=True,
        force_splits=True,
    )

    data_handler.load_data_loaders()
    print(data_handler.data_loaders["train"])
