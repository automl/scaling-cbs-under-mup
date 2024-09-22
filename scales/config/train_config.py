from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from litgpt.config import Config

# from litgpt.model import GPT
from litgpt.utils import num_parameters, parse_devices
from mup import get_shapes, load_base_shapes, make_base_shapes

from scales.config.base_config import BaseConfig
from scales.config.ConfigWrapper import ConfigWrapper
from scales.config.data_config import DataHandler, preprocess_wikitext
from scales.config.eval_config import EvalHandler
from scales.config.log_args import LoggingArgs
from scales.lr_utils import LRScheduler
from scales.model import GPT_Scales


def resolve_model_config(
    model_config: Config | None = None,
    model_config_path: Path | None = None,
    model_checkpoint_dir: Path | None = None,
    model_name: str | None = None,
) -> Config:
    """4 methods of loading a model configuration...

    Make sure this function always returns an initialized Config no matter what the train_config args are

    """
    if model_checkpoint_dir is not None:
        model_checkpoint_dir = Path(model_checkpoint_dir)

    model_config_path = (
        model_checkpoint_dir / "model_config.yaml"
        if model_config_path is None and model_checkpoint_dir is not None and model_checkpoint_dir.is_dir()
        else model_config_path
    )

    if model_config is None:
        # Setting up model configuration
        if model_config_path and model_name is None:
            config = Config.from_file(model_config_path)
        elif model_name and model_config_path is None:
            config = Config.from_name(model_name)
        elif model_config_path and model_name:
            raise ValueError("Only one of `model_name` or `model_config_path` can be set.")
        else:
            raise ValueError("Please specify `model_name` or `model_config_path`")
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
    devices: int = 1,
    deepseek_hparams: bool = True,
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
    if deepseek_hparams and (max_tokens is None or tokens_per_param is None) and max_train_steps:
        raise ValueError("When using `deepseek_hparams`, one should use either `max_tokens` or `tokens_per_param`")

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
        elif micro_batch_size and block_size:
            train_steps = int(max_tokens // (micro_batch_size * block_size * accumulation_iters * devices))
        else:
            raise ValueError(
                f"Either tokens_per_step="
                f"{tokens_per_step} or both batch_size={micro_batch_size} "
                f"and block_size={block_size} must be set with max_tokens={max_tokens}"
            )

    return train_steps


def get_mup_base_shape(target_config: Config | ConfigWrapper, base_scales: dict[str, int] | None) -> dict | None:
    base_config = ConfigWrapper.from_config(target_config)
    delta_config = deepcopy(base_config)
    if base_scales is not None:
        # Set the scale of base and delta config
        for name, base_scale in base_scales.items():
            setattr(base_config, name, base_scale)
            setattr(delta_config, name, base_scale * 2)
        base_config = ConfigWrapper.from_config(base_config.config)
        delta_config = ConfigWrapper.from_config(delta_config.config)

    base_shapes = get_shapes(GPT_Scales(base_config, mup_init=True))
    delta_shapes = get_shapes(GPT_Scales(delta_config, mup_init=True))

    return make_base_shapes(base_shapes, delta_shapes)


@dataclass
class TrainConfig(BaseConfig):
    """Configuration to specify a recipie for the model training. This class initializes all the necessary values used
    during the training based on the initialization arguments.

    Note:
        This object initializes a Config object which is an input to the GPT model.
        We never initialize or load model weights inside TrainConfig
    Note:
        The arguments that are not appended to the self.ignore_list are not allowed to change during the lifecycle
        of this object. This is because, those arguments are written into the yaml files when the config is saved,
        and loaded using those exact values again. Check true_weight_decay attribute for an example.
    Note:
        Avoid putting paths inside config objects as they are not reliable, and require to be reset for
         every training experiment.

    """

    micro_batch_size: int
    """The batch per iteration."""
    block_size: int
    """Max sequence length/context length/block size."""
    weight_decay: float
    """Weight Decay for AdamW optimizer."""
    max_val_steps: int
    """N of validation steps on validation data."""
    max_lr: float | None = None
    """The maximum Learning Rate."""
    accumulation_iters: int = 1
    """Number of accumulation iters per device."""
    devices: int | str = "auto"
    """The number of devices to be trained on."""

    # model config
    model_config: ConfigWrapper | Config | None = None
    """Config object for model config."""
    model_config_path: Path | None = None
    """Config Path for the Config object, ignored if model_config provided."""
    model_name: str | None = None
    """Model name to load from HF hub."""
    weight_init_type: Literal["plain", "scaled", "GPT-NeoX", "DeepSeek"] | None = None
    """Model weight initialization."""

    # LR scheduler
    cooldown_fraction: float | None = None
    """Fraction of steps to cooldown schedule."""
    warmup_fraction: float | None = None
    """Fraction of steps in the warmup schedule."""
    min_lr: float = 0.0
    """`min_lr` the scheduler can reach."""
    cosine_scheduler: bool = False
    """Cosine annealing scheduler."""
    scheduler_args: dict | None = None
    """Cosine annealing scheduler arugments."""

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
    z_loss_eps: float | None = None
    "Epsilon value for Z loss"

    # optimizer
    adam_beta_1: float = 0.9
    """Adam beta_1."""
    adam_beta_2: float = 0.95
    """Adam beta_2."""
    adam_eps: float = 1e-8
    """Adam epsilon."""
    independent_wd: bool = False
    "Whether to use independent weight decay during AdamW"

    # MuParam width
    mup_base_scales: dict[str, int] | int | None = None
    """Dict of scaling dimension to base scale."""
    mup_base_shape_path: str | Path | None = None
    """The path of the base model shape, ."""

    # DeepSeek hyperparameters
    deepseek_hparams: bool = False
    """Changes the learning rate, accumulation iters, and weight initialization to match deepseek's algorithm based on
    compute."""

    # logging details
    tracked_metrics: dict[str, int] | None = None
    global_log_step: int = 1

    # seeding
    seed: int = 444

    # checkpoint management
    load_state_path: Path | None = None
    """Path to load checkpoint, random states, etc.

    for continued training.

    """
    save_state_path: Path | None = None
    """Path to save checkpoint, random states, etc.

    for continued training.

    """
    save_state_every: int | None = None
    """Number of steps to save the state.

    If None, same as `validate every`

    """
    overwrite_state: bool = True
    """If True, overwrite the state, using the same filename.

    If False, append step to filename.

    """
    recovery_state: bool = False
    """Does not overwrite the latest saved checkpoint, but instead renames it as "lit_model_recovery.pth" and saves a
    new checkpoint (does this iteratively after every step), and automatically sets the model to load the recovered
    checkpoint instead of main.

    The reason for this addition is cases where we face failures during saving the model.

    """

    save_init_state: bool = True
    """Whether to save the initial state, especially the initialization weights."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_state_every = self.validate_every if self.save_state_every is None else self.save_state_every

        self.ignore_fields.extend(["model_config_path", "model_name"])
        self.model_config = resolve_model_config(
            self.model_config, self.model_config_path, self.load_state_path, self.model_name
        )
        # override model block_size
        self.model_config.block_size = self.block_size
        self.model_config = ConfigWrapper.from_config(self.model_config)
        self._mup_base_shape: dict | None = None

        self.trainable_params = num_parameters(GPT_Scales(self.model_config), requires_grad=True)
        self.devices = parse_devices(self.devices)

        if isinstance(self.devices, str):
            raise ValueError("`devices` is wrongly initialized and should be an in")

        if self.deepseek_hparams:
            model_scale = (
                72 * self.model_config.n_layer * (self.model_config.config.n_embd**2)
                + 12 * self.model_config.n_layer * self.model_config.d_model * self.model_config.block_size
            )
            if self.max_tokens:
                compute = model_scale * self.max_tokens
            elif self.tokens_per_param and self.trainable_params:
                compute = model_scale * self.tokens_per_param * self.trainable_params
            else:
                raise ValueError("An error has accured during DeepSeek's compute calculation")

            optim_deepseek_lr = 0.3119 * (compute**-0.125)
            optim_deepseek_effective_batch_size = 0.2920 * (compute**0.3271)

            self.max_lr = optim_deepseek_lr
            self.accumulation_iters = round(
                optim_deepseek_effective_batch_size
                / (self.devices * self.micro_batch_size * self.model_config.block_size)
            )

            # Just a check for when the rounding is too low
            if self.accumulation_iters <= 0:
                self.accumulation_iters = 1

        self.train_steps = resolve_train_steps(
            max_tokens=self.max_tokens,
            max_train_steps=self.max_train_steps,
            tokens_per_param=self.tokens_per_param,
            micro_batch_size=self.micro_batch_size,
            block_size=self.block_size,
            trainable_params=self.trainable_params,
            accumulation_iters=self.accumulation_iters,
            devices=self.devices,
            deepseek_hparams=self.deepseek_hparams,
        )

        if self.max_lr is None:
            raise ValueError("`max_lr` should not be `None`")

        self.lr_scheduler = LRScheduler(
            max_steps=self.train_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_frac=self.warmup_fraction,
            cool_down_frac=self.cooldown_fraction,
            scheduler_args=self.scheduler_args,
            cosine_scheduler=self.cosine_scheduler,
        )

        self.tracked_metrics = {} if self.tracked_metrics is None else self.tracked_metrics

        self.logging_args = LoggingArgs(
            tracked_metrics=self.tracked_metrics,
            global_log_step=self.global_log_step,
            log_dir=None,
        )

    @property
    def true_weight_decay(self) -> float:
        if self.independent_wd:
            return self.weight_decay / self.lr_scheduler.max_lr
        return self.weight_decay

    @property
    def mup_base_shape(self) -> dict | None:
        if self._mup_base_shape is not None:
            return self._mup_base_shape
        if self.mup_base_scales is None and self.mup_base_shape_path is not None:
            self._mup_base_shape = load_base_shapes(str(self.mup_base_shape_path))
            return self._mup_base_shape
        if isinstance(self.mup_base_scales, int):
            self.mup_base_scales = {"d_model": self.mup_base_scales}
        if isinstance(self.mup_base_scales, dict):
            self._mup_base_shape = get_mup_base_shape(self.model_config, self.mup_base_scales)

        return self._mup_base_shape

    @classmethod
    def from_yaml(cls, yaml_config: dict[str, Any], yaml_hook: Callable | None = None) -> TrainConfig:
        if yaml_hook is not None:
            yaml_config = yaml_hook(yaml_config)
        try:
            yaml_config["model_config"] = ConfigWrapper.from_yaml(yaml_config["model_config"])
        except TypeError:
            # Depending on if the train_config was saved with defaults or not
            # the model_config might have extra arguments
            yaml_config["model_config"] = ConfigWrapper.from_config(Config(**yaml_config["model_config"]))
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
            self.data_config = DataHandler.from_path(path=self.data_config_path)
        if self.train_config is None and self.train_config_path and self.train_config_path.exists():
            self.train_config = TrainConfig.from_path(path=self.train_config_path)
        if self.eval_config is None and self.eval_config_path and self.eval_config_path.exists():
            self.eval_config = EvalHandler.from_path(path=self.eval_config_path)

        assert self.data_config is not None
        assert self.train_config is not None
        self.data_config.block_size = self.train_config.block_size

    @classmethod
    def from_yaml(cls, yaml_config: dict[str, Any], yaml_hook: Callable | None = None) -> PipelineConfig:
        yaml_config["data_config"] = DataHandler.from_yaml(yaml_config["data_config"])
        yaml_config["train_config"] = TrainConfig.from_yaml(yaml_config["train_config"], yaml_hook)
        if yaml_config.get("eval_config"):
            yaml_config["eval_config"] = EvalHandler.from_yaml(yaml_config["eval_config"])
        else:
            yaml_config["eval_config"] = None
        return cls(**yaml_config)


if __name__ == "__main__":
    conf = TrainConfig(
        max_lr=0.001,
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
