from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from litdata.processing.functions import optimize
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset
from litgpt.tokenizer import Tokenizer

from scales.config.base_config import BaseConfig
from scales.config.utils import download_tokenizer, preprocess_wikitext, simple_filter, tokenize_wikitext


@dataclass
class DataHandler(BaseConfig):
    """DataHandler class for handling the dataset downloading, processing, and loading.
    
    This class defines a dataset and how it was processed, tokenized, split, optimized and saved. 
    When downloading, this class will create an YAML file in the target folder. 
    Later when the Dataset is requested again if the arguments of this class matches 
    the saved configuration in the YAML file the dataset will be loaded, otherwise, 
    it'll download the dataset again.

    WARNING: All fields not passed to the `self.ignore_fields` will be checked

    """
    # Dataset identifier for the HuggingFace Datasets
    hf_dataset_id: str
    
    # Dataset Subset name for the datasets with multiple subsets
    hf_data_subset_name: str
    
    # Preprocess Dataset before passing into tokenization, used in `Dataset.map(...)` function
    preprocess_fn: Callable[[dict], dict]
    
    # HuggingFace repository ID for the model which we will use the tokenizer of
    tokenizer_repo_id: str
    
    # HuggingFace dataset data_files
    hf_data_files: str | None = None
    
    # Filter function to be called on each dataset object immediately after loading
    filter_function: Callable[[dict], bool] = simple_filter

    # Tokenizer function to be used for the `optimize` function of `litdata.processing.functions`
    #   Takes in at least a single argument, which is the index (or list of indices) of the dataset
    #   and returns a tensor for the corresponding text.
    tokenizer_fn: Callable = tokenize_wikitext
    
    # Root folder which holds tokenizers, binaries, cache folders
    root_data_path: Path = Path(__file__).parent.parent.parent / "data"

    # Data splits to create or load from hf hub
    #  If any split other than the 'train' split doesn't exist,
    #  the loader will try to load all the splits and split them according to the `default_split_ratio`.
    #  If 'train' split is specified but not found then an exception will occur.
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])

    # Split ratios for data splits when either a split specified is not found on the hub
    # or `force_splits` flag is set
    default_split_ratio: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    
    # Forces the loader to load all the splits existing on the hub for the dataset and 
    # split them according to the `default_split_ratio`
    force_splits: bool = False
    
    # Forces to process the dataset again if set
    force_overwrite: bool = False

    # `tokenizer_fn` can take multiple arguments if their defaults are specified here
    tokenizer_fn_kwargs: dict[Any, Any] = field(default_factory=dict)
    
    # Seed for splitting datasets and shuffling
    seed: int = 42

    # Size of the already subsampled set
    #   e.g: 4M, 47M, used only for loading not for writing
    #   Note: valid `subsample_size`s are generated only after the 
    #   first time `self.write_subsamples` is called
    # subsample_size: str | None = None
    
    # Valid subsample size indices are: `0`, `1`, `2`
    #   `0` corresponds to the full parent dataset, `1` to 50%, `2` to 5%
    subsample_index: int = 0
    
    # Optimize the data loading for nlp datasets (Will be removed in the next iteration)
    nlp_dataset: bool = True

    # Block size for data loading (Will be removed in the next iteration)
    block_size: int = 2048

    # TODO: organize filter calls to be consistent

    def __post_init__(self) -> None:
        super().__post_init__()
        # ignore fields that doesn't affect dataset installing
        self.ignore_fields.append("force_overwrite")
        self.ignore_fields.append("subsample_index")
        self.ignore_fields.extend(["nlp_dataset", "block_size"])
        # ignore fields that change based on the environment
        self.ignore_fields.extend(
            ["access_internet", "tokenizer_root_path", "bin_data_path", "cache_dir", "binary_path"]
        )
        self.datasets: dict[str, StreamingDataset] = {}
        self.data_loaders: dict[str, StreamingDataLoader] = {}
        self.access_internet: bool = True

    @property
    def tokenizer_root_path(self) -> Path:
        """Root folder where to store all the downloaded tokenizers.
        """
        return self.root_data_path / "tokenizers"

    @property
    def bin_data_path(self) -> Path:
        """Root folder where to store all the tokenized datasets.
        """
        return self.root_data_path / "binaries"

    @property
    def cache_dir(self) -> Path:
        """Cache directory for huggingface datasets.
        """
        return self.root_data_path / "cache"

    @property
    def binary_path(self) -> Path:
        """Main/parent dataset path.
        """
        return self.bin_data_path / self.hf_dataset_id / self.hf_data_subset_name

    def _set_split_paths(self, subsample_mode: bool) -> None:
        if subsample_mode:
            # Get the first split from the subsample and other splits from the parent
            self.split_paths: list[Path] = [
                self.binary_path.parent
                / f"{self.hf_data_subset_name}_{self.subsample_sizes[self.subsample_index]}M"
                / self.splits[0]
            ] + [self.binary_path / split for split in self.splits[1:]]
        else:
            self.split_paths = [self.binary_path / split for split in self.splits]

    def _get_subsample_sizes(self) -> None:
        # masquerade as the parent dataset here
        self._set_split_paths(False)
        # Load datasets
        self.__convert_to_binary()

        for i, split in enumerate(self.splits):
            self.datasets[split] = self.get_dataset(
                binary_path=self.split_paths[i], nlp_dataset=self.nlp_dataset, block_size=self.block_size
            )
        # Load end
        dataset = self.datasets[self.splits[0]]
        n_tokens = len(dataset) * dataset[0].shape[0]
        sample_sizes: list[int] = []
        if n_tokens > 20e6:
            # subsample at 50% and 5%
            sample_sizes = [
                int((n_tokens // 2e6)),
                int((n_tokens // 20e6)),
            ]
        self.subsample_sizes: list[int] = [int(n_tokens // 1e6)] + sample_sizes

    def __check_exists(self) -> bool:
        return (
            not self.force_overwrite
            and all(path.exists() for path in self.split_paths)
            and self.to_dict() == self.load_yaml(self.binary_path)
        )

    def __convert_to_binary(self) -> None:
        if self.__check_exists():
            # Return if all folders for splits exists and serialized version of the config matches the yaml file
            warnings.warn(
                f"Dataset {self.hf_dataset_id} already exists at {self.binary_path}.\n"
                f"Quitting...\n"
                f"If you would like to prepare the dataset again set the `force_overwrite` flag."
            )
            return
        # Check if internet available
        if not self.access_internet:
            raise ValueError(
                f"Dataset can not be downloaded when the acccess_internet is {self.access_internet}, "
                f"please download the dataset to '{self.binary_path}' first,"
                f" with self.access_internet set to True"
            )

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
                **self.tokenizer_fn_kwargs,
            )
            # TODO: Script to find out optimal data prep batch size to speed up large datasets
            optimize(
                fn=token_fn,
                inputs=list(range(len(dataset))),
                output_dir=str(split_output_path),
                chunk_bytes="64MB",
                batch_size=1024,
            )
        self.write_yaml(self.binary_path, ignore_defaults=False)

    def __get_data_splits(self) -> dict[str, Dataset]:
        dataset_splits: dict[str, Any] = {}
        missing_splits: bool = False
        if not self.force_splits:
            for split in self.splits:
                try:
                    dataset: Dataset = (
                        load_dataset(
                            path=self.hf_dataset_id,
                            name=self.hf_data_subset_name,
                            split=split,
                            data_files=self.hf_data_files,
                            cache_dir=str(self.cache_dir),
                        )
                        .filter(self.filter_function)
                        .map(self.preprocess_fn)
                    )
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
                [
                    dataset_dict[split].filter(self.filter_function).map(self.preprocess_fn)
                    for split in list(dataset_dict.keys())
                ]
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

    def write_subsamples(self, nlp_dataset: bool = True, block_size: int = 2048) -> None:
        self._set_split_paths(False)
        # Load datasets
        self.__convert_to_binary()

        for i, split in enumerate(self.splits):
            self.datasets[split] = self.get_dataset(
                binary_path=self.split_paths[i], nlp_dataset=nlp_dataset, block_size=block_size
            )
        # Load end
        self._set_split_paths(True)
        dataset = self.datasets[self.splits[0]]

        for sample_size in self.subsample_sizes[1:]:
            subsample_out_path = (
                self.bin_data_path
                / self.hf_dataset_id
                / f"{self.hf_data_subset_name}_{int(sample_size)}M"
                / self.splits[0]
            )
            if (
                not self.force_overwrite
                and subsample_out_path.exists()
                and self.to_dict() == self.load_yaml(subsample_out_path)
            ):
                continue
            block_size = dataset[0].shape[0]
            max_input = int(1e6 * sample_size // block_size)
            inputs = list(range(max_input))
            # print(f"Inputs: {inputs}")
            print(sample_size)
            print(max_input)

            def subsample_(indices: list[int] | int, dataset: StreamingDataset) -> torch.Tensor:
                # TODO: not very optimized
                if isinstance(indices, int):
                    yield dataset[indices]
                else:
                    yield torch.cat([dataset[i] for i in indices])

            compress_fn = partial(subsample_, dataset=dataset)

            optimize(
                fn=compress_fn,
                inputs=inputs,
                output_dir=str(subsample_out_path),
                chunk_bytes="64MB",
                batch_size=1024,
            )
            self.write_yaml(subsample_out_path.parent, ignore_defaults=False)

    def get_dataset(self, binary_path: Path, nlp_dataset: bool = True, block_size: int = 2048) -> StreamingDataset:
        item_loader = None
        if nlp_dataset:
            from litdata.streaming.item_loader import TokensLoader

            item_loader = TokensLoader(block_size=block_size + 1)

        return StreamingDataset(input_dir=str(binary_path), item_loader=item_loader, shuffle=True, seed=self.seed)

    def load_datasets(
        self,
        nlp_dataset: bool = True,
        block_size: int = 2048,
    ) -> None:
        """Loads the `StreamingDatasets` into `self.datasets` dict, keys are `splits`

        Args:
            nlp_dataset: Use optimized data reading for the `StreamingDataset` if the dataset is an nlp dataset.
            block_size: Block size in each batch.

        """
        # TODO: write a helper function to combine multiple datasets with `CombinedStreamingDataset` if necessary
        if self.subsample_index:
            self._get_subsample_sizes()
            self._set_split_paths(True)
            if not self.__check_exists():
                # if any path doesn't exist generate subsample sets first
                self.write_subsamples(nlp_dataset=self.nlp_dataset, block_size=self.block_size)
        else:
            self._set_split_paths(False)
            self.__convert_to_binary()

        for i, split in enumerate(self.splits):
            self.datasets[split] = self.get_dataset(
                binary_path=self.split_paths[i], nlp_dataset=nlp_dataset, block_size=block_size
            )

    def load_data_loaders(
        self,
        nlp_dataset: bool = True,
        batch_size: int = 64,
        block_size: int = 2048,
        num_workers: int = 1,
        access_internet: bool = True,
    ) -> None:
        """Loads the `StreamingDataLoaders` into `self.data_loaders` dict, keys are `splits`

        Args:
            nlp_dataset: Use optimized data reading for the `StreamingDataset` if the dataset is an nlp dataset.
            batch_size: Batch size for `StreamingDataLoader` (same for all splits)
            block_size: Block size in each batch.
            num_workers: Number of workers for `StreamingDataLoader`
            access_internet: Enable data downloading

        """
        self.access_internet = access_internet
        self.load_datasets(nlp_dataset=nlp_dataset, block_size=block_size)
        for split in self.splits:
            self.data_loaders[split] = StreamingDataLoader(
                dataset=self.datasets[split], batch_size=batch_size, num_workers=num_workers
            )


if __name__ == "__main__":
    data_handler = DataHandler(
        hf_dataset_id="wikitext",
        hf_data_subset_name="wikitext-103-v1",
        tokenizer_repo_id="openai-community/gpt2",
        preprocess_fn=preprocess_wikitext,
        force_overwrite=False,
        force_splits=True,
        subsample_index=0,
    )
    # data_handler = DataHandler.from_path(Path("/data/binaries/wikitext/wikitext-2-v1"))
    # data_handler.write_subsamples()
    # TODO: improve subsampling interface
    data_handler.load_data_loaders(access_internet=True)
    s = 0
    print(len(data_handler.data_loaders["train"]))
    for batch in data_handler.data_loaders["train"]:
        s_ = batch.shape[0] * batch.shape[1]
        s += s_
    print(s)

    s = 0
    print(len(data_handler.data_loaders["test"]))
    for batch in data_handler.data_loaders["test"]:
        s_ = batch.shape[0] * batch.shape[1]
        s += s_
    print(s)
