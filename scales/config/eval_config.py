# TODO: Get eval config
#   1. Non-finetuning tasks from lm_eval
#   2.1 Finetuning tasks loaded from dataset_configs
#       2.1.1 generation datasets
#       2.1.2 label prediction datasets?
#       2.1.3 Dataset filter + map functions
#   2.2. Finetuning params
#       2.2.1. Which/How many layers to freeze
#       2.2.2 How long to finetune
#       2.2.3 How to finetune (generation / label prediction)

# TODO: Given a dataHandler, finetune (simple train) on it and evaluate val results
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning as L

from scales.config.base_config import BaseConfig
from scales.eval_utils import convert_and_evaluate


@dataclass
class EvalHandler(BaseConfig):
    model_dir: str | Path
    """Checkpoint directory of the trained litgpt model."""
    tokenizer_dir: str | Path
    """Directory containing the tokenizer specifications."""
    lm_eval_tasks: str = "mmlu_professional_law"
    """Comma separated list of `lm_eval` `task_id`s.

    Print all possible task ids by running:
    ```python
    from lm_eval.tasks import TaskManager
    taskm = TaskManager()
    print("\n".join(taskm.task_index.keys()))
    ```

    """
    seed: int = 1234
    """Random seed."""
    num_fewshot: int | None = None
    """Number of examples in few-shot context."""
    # is_finetuning: bool
    data_handler_path: Path | None = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.output_dir = self.model_dir / "evaluate"
        self.tokenizer_dir = Path(self.tokenizer_dir)

    def evaluate(self) -> None:
        fabric = L.Fabric(devices="auto", strategy="auto")
        convert_and_evaluate(
            fabric=fabric,
            checkpoint_dir=self.model_dir,  # type: ignore
            tokenizer=self.tokenizer_dir,
            tasks=self.lm_eval_tasks,
            out_dir=self.output_dir,
        )
        self.write_yaml(self.output_dir)


if __name__ == "__main__":
    eval_ = EvalHandler(
        model_dir=Path(__file__).parent.parent / "output",
        tokenizer_dir=Path(__file__).parent.parent / Path("data/tokenizers/openai-community/gpt2"),
    )
    eval_.evaluate()
