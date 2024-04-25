from __future__ import annotations

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Literal

import neps
import torch
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.data import TinyLlama
from litgpt.data.base import DataModule
from litgpt.eval.evaluate import convert_and_evaluate
from litgpt.pretrain import setup


class MISSING:
    pass


def get_args(caller: Any, kwargs: dict, arg_name: str, __dynamic_default: MISSING | Any = MISSING()) -> Any:
    """Get parameter value from either kwargs or the DEFAULTS dictionary. Will try to fall back on __default value if
    provided but is not advised.

    Args:
    ----
        caller: Class instance calling this function
        kwargs (dict): Arguments dictionary
        arg_name (str): Parameter name
        __dynamic_default: Default value for dynamic defaults (if needed)

    """
    try:
        return kwargs.pop(arg_name, caller.DEFAULTS[arg_name])
    except KeyError as e:
        if isinstance(__dynamic_default, MISSING):
            raise ValueError(
                f"parameter {arg_name} is not specified in the {type(caller)}.DEFAULTS dictionary "
                f"or the search-space. Either add it to the pipeline search-space or "
                f"add it to the constants dictionary or specify a __dynamic_default value"
            ) from e

        caller.logger.info(f"Using __dynamic_default value of {__dynamic_default} for {arg_name}")
        return __dynamic_default


class PreTrainLitGPT:
    # TODO: maybe move all defaults to a yaml config?
    # Add all defaults here, even the ones in the search-space
    DEFAULTS = dict(  # noqa: C408
        # TrainArgs
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),
        learning_rate=4e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
        # EvalArgs
        interval=1000,
        max_iters=100,
        # Model Config
        scale_embeddings=False,
        block_size=4096,
        vocab_size=50254,
        padding_multiple=512,
        padded_vocab_size=None,
        n_layer=16,
        n_head=32,
        head_size=None,
        n_embd=4096,
        rotary_percentage=0.25,
        parallel_residual=True,
        bias=True,
        lm_head_bias=False,
        n_query_groups=None,
        shared_attention_norm=False,
        norm_class_name="LayerNorm",
        norm_eps=1e-5,
        mlp_class_name="GptNeoxMLP",
        gelu_approximate="none",
        intermediate_size=None,
        rope_condense_ratio=1,
        rope_base=10000,
        n_expert=0,
        n_expert_per_token=0,
    )

    def __init__(
        self,
        data_module: DataModule,
        tokenizer_dir: Path | str,
        seed: int = 42,
        logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
        logger: Logger = getLogger(),
        evaluator_defaults: dict | None = None,
        update_defaults: dict | None = None,
    ):
        self.evaluator = EvaluateLitGPT(seed=seed, update_defaults=evaluator_defaults, logger=logger)
        # TODO: add DataModule support for all pre-training datasets
        self.data_module = data_module
        self.tokenizer_dir = tokenizer_dir
        self.seed = seed
        self.logger_name = logger_name
        self.logger = logger

        if update_defaults is not None:
            self.DEFAULTS.update(update_defaults)

    def __call__(self, pipeline_directory: str | Path, previous_pipeline_directory: str | Path, **kwargs: int) -> dict:
        """Generic run_pipeline function to be converted into a proper piece of code later."""
        pipeline_directory = Path(pipeline_directory) if isinstance(pipeline_directory, str) else pipeline_directory
        previous_pipeline_directory = (
            Path(previous_pipeline_directory)
            if isinstance(previous_pipeline_directory, str)
            else previous_pipeline_directory
        )

        out_dir = pipeline_directory / "litgpt"

        initial_checkpoint_dir = None
        resume = False
        if previous_pipeline_directory is not None:
            initial_checkpoint_dir = previous_pipeline_directory / "litgpt" / "final"
            resume = True

        train_args = TrainArgs(
            save_interval=get_args(self, kwargs, "save_interval"),
            log_interval=get_args(self, kwargs, "log_interval"),
            global_batch_size=get_args(self, kwargs, "global_batch_size"),
            micro_batch_size=get_args(self, kwargs, "micro_batch_size"),
            max_tokens=get_args(self, kwargs, "max_tokens"),
            learning_rate=get_args(self, kwargs, "learning_rate"),
            weight_decay=get_args(self, kwargs, "weight_decay"),
            beta1=get_args(self, kwargs, "beta1"),
            beta2=get_args(self, kwargs, "beta2"),
            max_norm=get_args(self, kwargs, "max_norm"),
            min_lr=get_args(self, kwargs, "min_lr"),
            lr_warmup_steps=get_args(self, kwargs, "lr_warmup_steps"),
            tie_embeddings=get_args(self, kwargs, "tie_embeddings"),
        )

        eval_args = EvalArgs(interval=get_args(self, kwargs, "interval"), max_iters=get_args(self, kwargs, "max_iters"))

        model_config = Config(
            name=pipeline_directory.name,
            hf_config={},
            scale_embeddings=get_args(self, kwargs, "scale_embeddings"),
            block_size=get_args(self, kwargs, "block_size"),
            vocab_size=get_args(self, kwargs, "vocab_size"),
            padding_multiple=get_args(self, kwargs, "padding_multiple"),
            padded_vocab_size=get_args(self, kwargs, "padded_vocab_size"),
            n_layer=get_args(self, kwargs, "n_layer"),
            n_head=get_args(self, kwargs, "n_head"),
            head_size=get_args(self, kwargs, "head_size"),
            n_embd=get_args(self, kwargs, "n_embd"),
            rotary_percentage=get_args(self, kwargs, "rotary_percentage"),
            parallel_residual=get_args(self, kwargs, "parallel_residual"),
            bias=get_args(self, kwargs, "bias"),
            lm_head_bias=get_args(self, kwargs, "lm_head_bias"),
            n_query_groups=get_args(self, kwargs, "n_query_groups"),
            shared_attention_norm=get_args(self, kwargs, "shared_attention_norm"),
            norm_class_name=get_args(self, kwargs, "norm_class_name"),
            norm_eps=get_args(self, kwargs, "norm_eps"),
            mlp_class_name=get_args(self, kwargs, "mlp_class_name"),
            gelu_approximate=get_args(self, kwargs, "gelu_approximate"),
            intermediate_size=get_args(self, kwargs, "intermediate_size"),
            rope_condense_ratio=get_args(self, kwargs, "rope_condense_ratio"),
            rope_base=get_args(self, kwargs, "rope_base"),
            n_expert=get_args(self, kwargs, "n_expert"),
            n_expert_per_token=get_args(self, kwargs, "n_expert_per_token"),
        )

        val_ppl = setup(
            model_config=model_config,
            out_dir=out_dir,
            initial_checkpoint_dir=initial_checkpoint_dir,
            resume=resume,
            data=self.data_module,
            train=train_args,
            eval=eval_args,
            devices="auto",
            tokenizer_dir=self.tokenizer_dir,
            logger_name=self.logger_name,
            seed=self.seed,
        )

        results = self.evaluator.evaluate(
            checkpoint_dir=pipeline_directory / "litgpt" / "final",
            out_dir=None,
            tasks=get_args(self.evaluator, kwargs, "tasks"),
            num_fewshots=get_args(self.evaluator, kwargs, "num_fewshots"),
            batch_size=get_args(self.evaluator, kwargs, "batch_size"),
            limit=get_args(self.evaluator, kwargs, "limit"),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        print(results)
        return {"loss": val_ppl}


class EvaluateLitGPT:
    """Evaluation options for LitGPT models through `lm_eval`.

    Might be useful if we ever need to search the evaluation metrics space as well. Otherwise, a simple interface to
    specify the evaluation metrics.

    """

    DEFAULTS = dict(  # noqa: C408
        tasks=("hellaswag", "truthfulqa_mc2", "mmlu"), num_fewshots=None, batch_size=1, limit=None
    )

    def __init__(self, seed: int = 42, logger: Logger = getLogger(), update_defaults: dict | None = None):
        self.seed = seed
        self.logger = logger

        if update_defaults is not None:
            self.DEFAULTS.update(update_defaults)

    @staticmethod
    def all_tasks() -> None:
        """Get the list of all tasks available (len > 2000) we can use below code in a console."""
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))

    def evaluate(
        self,
        checkpoint_dir: Path,
        tasks: list,
        out_dir: str | None,
        # force_conversion: bool,
        num_fewshots: int | None,
        batch_size: int,
        device: str | None,
        limit: float,
    ) -> Any:
        """Pass arguments to litgpt function and parse results into info_dict."""
        # TODO: maybe delete this and convert the class into a dataclass?
        if tasks is not None and len(tasks) > 0:
            tasks = ",".join(tasks)  # type: ignore
            return convert_and_evaluate(
                checkpoint_dir, tasks, out_dir, False, num_fewshots, batch_size, device, limit, self.seed
            )
            # TODO: Parse results after this
        return None


if __name__ == "__main__":
    # How a neps run might look like
    # Ideally we would only change the pipeline_space and default overrides (if needed) through a yaml config
    # Then we can have different yaml files for different neps processes
    pipeline_space = {
        "n_layer": neps.IntegerParameter(lower=8, upper=32, default=16),
        "n_head": neps.IntegerParameter(lower=1, upper=64, default=32),
        # "max_tokens": neps.IntegerParameter(lower=1e6, upper=3e12, log=True)
    }

    objective = PreTrainLitGPT(data_module=TinyLlama(), tokenizer_dir="tokenizer/path/to/tinyllama/tokenizer", seed=42)

    # NOT RUNNABLE YET
    neps.api.run(
        run_pipeline=objective,
        pipeline_space=pipeline_space,
        root_directory="results/pretrain_run",
        max_evaluations_total=50,
    )
