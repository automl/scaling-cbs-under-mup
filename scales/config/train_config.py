from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from warnings import warn

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import num_parameters, parse_devices

from scales.args import LoggingArgs
from scales.config.base_config import BaseConfig
from scales.config.data_config import DataHandler, preprocess_wikitext
from scales.config.eval_config import EvalHandler
from scales.lr_utils import LRScheduler


def resolve_model_config(
    model_config: Config | None = None,
    checkpoint_dir: Path | None = None,
    model_config_path: Path | None = None,
    model_name: str | None = None,
) -> Config:
    """4 methods of loading a model configuration..."""
    if model_config is None:
        if checkpoint_dir is None:
            # Setting up model configuration
            if model_config_path and model_name is None:
                config = Config.from_file(model_config_path)
            elif model_name and model_config_path is None:
                config = Config.from_name(model_name)
            elif model_config_path and model_name:
                raise ValueError("Only one of `model_name` or `model_config` can be set.")
            else:
                raise ValueError("Please specify `model_name` or `model_config_file`")

        else:
            if model_config_path or model_name:
                warn(
                    "The configuration yaml in the loaded directory will be used "
                    "`model_config_file` and `model_name` are ignored"
                )
            model_path = Path(checkpoint_dir / "model_config.yaml")
            config = Config.from_file(model_path)
    else:
        return model_config

    return config


def resolve_train_steps(
    max_tokens: int | None = None,
    max_train_steps: int | None = None,
    tokens_per_param: int | None = None,
    micro_batch_size: int | None = None,
    block_size: int | None = None,
    trainable_params: int | None = None,
    tokens_per_step: int | None = None,
    accumulation_iters: int = 1,
    devices: int | str = "auto",
) -> int:
    """3 ways of computing train_steps...

    available settings:
        max_train_steps,
        tokens_per_param, trainable_params
        max_tokens, tokens_per_step
        max_tokens, batch_size, block_size

    """
    if max_train_steps is None and tokens_per_param is None and max_tokens is None:
        raise ValueError("One of max_train_steps, tokens_per_param, max_tokens must be set.")
    if (tokens_per_param and max_tokens) or (max_train_steps and max_tokens) or (max_train_steps and tokens_per_param):
        raise ValueError(
            f"Only one of max_train_steps={max_train_steps}, "
            f"tokens_per_param={tokens_per_param}, max_tokens={max_tokens} can be set."
        )

    if max_train_steps:
        train_steps = max_train_steps

    if tokens_per_param:
        if trainable_params:
            max_tokens = trainable_params * tokens_per_param
        else:
            raise ValueError(
                f"when tokens_per_param={tokens_per_param} is set, trainable_params={trainable_params} "
                f"must also be set."
            )

    devices = parse_devices(devices=devices)

    if max_tokens:
        if tokens_per_step:
            train_steps = int(max_tokens // tokens_per_step)
        elif micro_batch_size and block_size:
            train_steps = int(max_tokens // (micro_batch_size * block_size * accumulation_iters * devices))
        else:
            raise ValueError(
                f"Either tokens_per_step="
                f"{tokens_per_step} or both batch_size={micro_batch_size} "
                f"and block_size={block_size} must be set with max_tokens={max_tokens}"
            )

    return train_steps


def resolve_scheduler_params(
    init_lr: float,
    max_train_steps: int,
    min_lr: float = 0,
    warmup_steps: int | None = None,
    main_steps: int | None = None,
    cooldown_steps: int | None = None,
    torch_scheduler: str | None = None,
    torch_scheduler_args: dict | None = None,
) -> LRScheduler:
    warmup_steps = 0 if warmup_steps is None else warmup_steps
    cooldown_steps = 0 if cooldown_steps is None else cooldown_steps
    main_steps = max_train_steps - warmup_steps - cooldown_steps if main_steps is None else main_steps

    assert (
        warmup_steps + cooldown_steps + main_steps == max_train_steps
    ), f"LR args doesn't match max_train_steps = {max_train_steps} != {warmup_steps} + {cooldown_steps} + {main_steps}"

    end_warmup_step = warmup_steps
    end_decay_step = warmup_steps + main_steps
    end_cooldown_step = warmup_steps + main_steps + cooldown_steps

    # TODO: write an adapter for all torch.optim.LRScheduler classes
    if torch_scheduler and torch_scheduler == "CosineAnnealingLR":
        torch_scheduler_args = {} if torch_scheduler_args is None else torch_scheduler_args
        torch_scheduler_args["T_max"] = main_steps

    return LRScheduler(
        init_lr=init_lr,
        min_lr=min_lr,
        end_decay_step=end_decay_step,
        end_warmup_step=end_warmup_step,
        end_cooldown_step=end_cooldown_step,
        torch_scheduler=torch_scheduler,
        torch_scheduler_args=torch_scheduler_args,
    )


@dataclass
class TrainConfig(BaseConfig):
    init_lr: float
    """Initial Learning Rate."""
    micro_batch_size: int
    """The batch per iteration."""
    block_size: int
    """Max sequence length/context length/block size."""
    weight_decay: float
    """Weight Decay for AdamW optimizer."""
    max_val_steps: int
    """N of validation steps on validation data."""
    accumulation_iters: int = 1
    """Number of accumulation iters per device."""
    devices: int | str = "auto"
    """The number of devices to be trained on."""

    # model config
    model_config: Config | None = None
    """Config object for model config."""
    model_config_path: Path | None = None
    """Config Path for the Config object, ignored if model_config provided."""
    model_checkpoint_dir: Path | None = None
    """Checkpoint directory for a trained model, ignored if model_config provided.

    NOTE: not used for loading pre-trained models. only for loading Config object

    """
    model_name: str | None = None
    """Model name to load from HF hub."""

    # LR scheduler
    n_main_steps: int | None = None
    """Number of steps in the torch main/decay schedule."""
    n_warmup_steps: int | None = None
    """Number of steps in the warmup schedule."""
    n_cooldown_steps: int | None = None
    """Number of steps in the cooldown schedule."""
    min_lr: float = 0
    """`min_lr` the scheduler can reach."""
    torch_scheduler: str | None = None
    """Torch type scheduler defined in a string."""
    torch_scheduler_args: dict | None = None
    """All torch scheduler arguments."""

    # training length
    max_train_steps: int | None = None
    """Max training steps to train for."""
    tokens_per_param: int | None = None
    """Used to calculate train_steps if train_steps not provided."""
    max_tokens: int | None = None

    # train details
    clip_max_norm: int | None = None
    clip_max_val: float | None = None
    validate_every: int = 5
    """Number of steps after which to validate the model."""

    # logging details
    tracked_metrics: list[str] | None = None
    log_step: int = 5
    log_dir: str | Path | None = None

    # seeding
    seed: int = 444

    # checkpoint management
    load_state_path: Path | None = None
    """Path to load checkpoint, random states, etc. for continued training."""
    save_state_path: Path | None = None
    """Path to save checkpoint, random states, etc. for continued training."""
    save_state_every: int | None = None
    """Number of steps to save the state. If None, same as `validate every`"""
    overwrite_state: bool = True
    """If True, overwrite the state, using the same filename. If False, append step to filename."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_state_every = (
           self.validate_every if self.save_state_every is None else self.save_state_every
        )

        self.ignore_fields.extend(["model_config_path", "model_checkpoint_dir", "model_name"])
        self.model_config = resolve_model_config(
            self.model_config, self.model_checkpoint_dir, self.model_config_path, self.model_name
        )
        # override model block_size
        self.model_config.block_size = self.block_size

        self.trainable_params = num_parameters(GPT(self.model_config), requires_grad=True)

        self.train_steps = resolve_train_steps(
            max_tokens=self.max_tokens,
            max_train_steps=self.max_train_steps,
            tokens_per_param=self.tokens_per_param,
            micro_batch_size=self.micro_batch_size,
            block_size=self.block_size,
            trainable_params=self.trainable_params,
            accumulation_iters=self.accumulation_iters,
            devices=self.devices,
        )

        self.lr_scheduler = resolve_scheduler_params(
            init_lr=self.init_lr,
            max_train_steps=self.train_steps,
            min_lr=self.min_lr,
            cooldown_steps=self.n_cooldown_steps,
            main_steps=self.n_main_steps,
            warmup_steps=self.n_warmup_steps,
            torch_scheduler=self.torch_scheduler,
            torch_scheduler_args=self.torch_scheduler_args,
        )

        self.tracked_metrics = [] if self.tracked_metrics is None else self.tracked_metrics

    @classmethod
    def from_yaml(cls, yaml_config: dict[str, Any]) -> TrainConfig:
        yaml_config["model_config"] = Config(**yaml_config["model_config"])
        return cls(**yaml_config)


@dataclass
class PipelineConfig(BaseConfig):
    data_config_path: Path | None = None
    train_config_path: Path | None = None
    eval_config_path: Path | None = None

    data_config: DataHandler | None = None
    train_config: TrainConfig | None = None
    eval_config: EvalHandler | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.data_config is None and self.data_config_path and self.data_config_path.exists():
            self.data_config = DataHandler.from_path(path=self.data_config_path)  # type: ignore
        if self.train_config is None and self.train_config_path and self.train_config_path.exists():
            self.train_config = TrainConfig.from_path(path=self.train_config_path)  # type: ignore
        if self.eval_config is None and self.eval_config_path and self.eval_config_path.exists():
            self.eval_config = EvalHandler.from_path(path=self.eval_config_path)  # type: ignore

        assert self.data_config is not None
        assert self.train_config is not None
        self.data_config.block_size = self.train_config.block_size

    @classmethod
    def from_yaml(cls, yaml_config: dict[str, Any]) -> PipelineConfig:
        yaml_config["data_config"] = DataHandler.from_yaml(yaml_config["data_config"])
        yaml_config["train_config"] = TrainConfig.from_yaml(yaml_config["train_config"])
        if yaml_config.get("eval_config"):
            yaml_config["eval_config"] = EvalHandler.from_yaml(yaml_config["eval_config"])
        else:
            yaml_config["eval_config"] = None
        return cls(**yaml_config)


if __name__ == "__main__":
    conf = TrainConfig(
        init_lr=0.001,
        micro_batch_size=1,
        block_size=1028,
        weight_decay=0.001,
        max_val_steps=2,
        model_config_path=Path(
            "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/model.yaml"
        ),
        max_train_steps=200,
    )
    conf.write_yaml(
        output_dir=Path("/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output")
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
    data_handler.load_data_loaders()
    config = TrainConfig.from_path(
        path=Path("/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output")
    )
    print(config)

    c = PipelineConfig(
        data_config_path=Path(
            "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/data/binaries/wikitext/wikitext-2-v1/DataHandler.yaml"
        ),
        train_config_path=Path(
            "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output/TrainConfig.yaml"
        ),
        eval_config_path=Path(
            "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output/evaluate/EvalHandler.yaml"
        ),
    )
    print(c)

    c.write_yaml(Path("/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output"))

# class PipelineConfig:
#     def __init__(self, DataConfig, TrainConfig, EvalConfig):
