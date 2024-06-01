from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Type
from warnings import warn

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.utils import num_parameters
from lr_utils import BaseLR, ConstantLR

from scales.config_utils import BaseConfig


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
    batch_size: int | None = None,
    block_size: int | None = None,
    trainable_params: int | None = None,
    tokens_per_step: int | None = None,
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

    if max_tokens:
        if tokens_per_step:
            train_steps = int(max_tokens // tokens_per_step)
        elif batch_size and block_size:
            train_steps = int(max_tokens // (batch_size * block_size))
        else:
            raise ValueError(
                f"Either tokens_per_step="
                f"{tokens_per_step} or both batch_size={batch_size} "
                f"and block_size={block_size} must be set with max_tokens={max_tokens}"
            )

    return train_steps


def resolve_scheduler_params(
    init_lr: float,
    lr_schedule_class: Type[BaseLR],
    min_lr: float | None = None,
    max_warmup_steps: int | None = None,
    start_decay_at_step: int | None = None,
    max_decay_steps: int | None = None,
) -> BaseLR:
    # Get this function args into a dict
    kwargs = {**locals()}
    _ = kwargs.pop("lr_schedule_class")

    if issubclass(lr_schedule_class, BaseLR):
        params = signature(lr_schedule_class.__init__).parameters
    else:
        raise ValueError("lr_schedule_class must be a subclass of BaseLR")

    param_dict = {}
    for key, val in kwargs.items():
        if key in params:
            # NOTE: passing `None` will trigger default value for that parameter
            param_dict[key] = params[key].default if kwargs[key] is None else kwargs[key]
            if param_dict[key] is Parameter.empty:
                raise TypeError(
                    f"Missing the value for a required positional argument: {key}.\n given arguments: {kwargs}"
                )

    return lr_schedule_class(**param_dict)


@dataclass
class TrainConfig(BaseConfig):
    init_lr: float
    batch_size: int
    block_size: int
    weight_decay: float
    max_val_steps: int

    # model config
    model_config: Config | None = None
    model_config_path: Path | None = None
    model_checkpoint_dir: Path | None = None
    model_name: str | None = None

    # LR scheduler
    lr_schedule_class: Type[BaseLR] = ConstantLR
    start_decay_at_step: int | None = None
    max_decay_steps: int | None = None
    max_warmup_steps: int | None = None
    min_lr: float | None = None

    # training length
    train_steps: int | None = None
    tokens_per_param: int | None = None
    max_tokens: int | None = None
    force_unique_tokens: bool = False

    # train details
    max_norm: int | None = None
    clip_val: float | None = None

    tracked_metrics: list[str] | None = None

    seed: int = 444

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ignore_fields.extend(["model_config_path", "model_checkpoint_dir", "model_name"])
        self.model_config = resolve_model_config(
            self.model_config, self.model_checkpoint_dir, self.model_config_path, self.model_name
        )
        # override model block_size
        self.model_config.block_size = self.block_size

        self.trainable_params = num_parameters(GPT(self.model_config), requires_grad=True)

        self.train_steps = resolve_train_steps(
            max_tokens=self.max_tokens,
            max_train_steps=self.train_steps,
            tokens_per_param=self.tokens_per_param,
            batch_size=self.batch_size,
            block_size=self.block_size,
            trainable_params=self.trainable_params,
        )

        self.lr_scheduler = resolve_scheduler_params(
            init_lr=self.init_lr,
            lr_schedule_class=self.lr_schedule_class,
            min_lr=self.min_lr,
            max_warmup_steps=self.max_warmup_steps,
            start_decay_at_step=self.start_decay_at_step,
            max_decay_steps=self.max_decay_steps,
        )


if __name__ == "__main__":
    conf = TrainConfig(
        init_lr=0.001,
        batch_size=1,
        block_size=1028,
        weight_decay=0.001,
        max_val_steps=2,
        model_config_path=Path(
            "/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/model.yaml"
        ),
        train_steps=200,
    )
    conf.write_yaml(
        output_dir=Path("/home/samir/Desktop/Projects/HiWi-AutoML/Thesis/scaling_all_the_way/examples/output")
    )


# class PipelineConfig:
#     def __init__(self, DataConfig, TrainConfig, EvalConfig):
