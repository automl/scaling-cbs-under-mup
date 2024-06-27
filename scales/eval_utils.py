from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table

from scales.config.ConfigWrapper import ConfigWrapper
from scales.utils import load_checkpoint


def convert_and_evaluate(
    fabric: L.Fabric,
    checkpoint_dir: Path,
    tokenizer: str | Path | Tokenizer,
    tasks: str | None = None,
    out_dir: Path | None = None,
    num_fewshot: int | None = None,
    batch_size: int = 1,
    device: str | None = None,
    limit: float | None = None,
    seed: int = 1234,
) -> None:
    """Convert a LitGPT model and run the LM Evaluation Harness.

    Args:
        fabric: lightning.Fabric module to load the model state
        checkpoint_dir: Directory where the `lit_model.pth` and tokenizer files are located.
        out_dir: Directory in which to save the converted checkpoints for evaluation.
            Saves to `checkpoint_dir`/evaluate by default.
        tasks: CSV of task names to evaluate. Example: "hellaswag,truthfulqa_mc2,mmlu"
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        seed: Random seed.

    """

    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `litgpt evaluate | grep task_name`."
        )
        return

    checkpoint_dir = Path(checkpoint_dir)

    if isinstance(tokenizer, (Path, str)):
        tokenizer = Path(tokenizer)
        tokenizer = Tokenizer(checkpoint_dir=tokenizer)

    out_dir = checkpoint_dir / "evaluate" if out_dir is None else Path(out_dir)

    out_dir.mkdir(exist_ok=True, parents=True)
    save_filepath = out_dir / Path("results.json")

    state, _ = load_checkpoint(fabric, checkpoint_dir)
    model = GPT(ConfigWrapper.from_path(checkpoint_dir / "model_config.yaml").config)
    model.load_state_dict(state_dict=state["model"])
    model = fabric.setup(model)
    model.eval()

    lm = LMGPT(model, tokenizer, device=device)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    prepare_results(results, save_filepath)


class LMGPT(TemplateLM):
    # TODO: validate implementation with a small Pythia test
    def __init__(self, model: GPT, tokenizer: Tokenizer, device: torch.device | str | None = None):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id

    def tok_encode(self, string: str, bos: bool | None = None, eos: bool = False, max_length: int = -1) -> torch.Tensor:
        return self.tokenizer.encode(string, device=self.device, bos=bos, eos=eos, max_length=max_length)

    def _encode_pair(self, context: str, continuation: str) -> tuple[torch.Tensor, torch.Tensor]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = (
                    torch.tensor([self.eot_token_id], dtype=torch.int, device=self.device),
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], torch.Tensor, torch.Tensor]], **kwargs: Any
    ) -> list[tuple[float, bool]]:
        # No Padding/batching or optimization for now
        res = []
        for idx, (string_pair, context_enc, cont_enc) in enumerate(requests):
            # reshape to add batch dimension, truncate to max context size, convert type
            model_input = (
                torch.cat([context_enc, cont_enc])[None, -(self.model.config.block_size + 1) : -1].contiguous().long()
            )
            logits = self.model(model_input)

            log_prob = F.log_softmax(logits.clone().detach(), dim=-1)
            # print(log_prob.device)
            logit = log_prob[-len(cont_enc) :]
            # dim: len(cont_enc)
            greedy_tokens = logit.argmax(dim=-1)

            max_equal = (greedy_tokens == cont_enc).all()

            cont_enc = cont_enc.unsqueeze(0)
            cont_enc = cont_enc.type(dtype=torch.int64)

            logit = torch.gather(logit, 2, cont_enc.unsqueeze(-1)).squeeze(-1)

            answer = (float(logit.sum()), bool(max_equal))

            res.append(answer)

        return res

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        # TODO: This is the wrong way of doing this, correct later
        res = []
        for string, dict_args in [req.args for req in requests]:
            model_input = self.tokenizer.encode(string).unsqueeze(0)
            # dim: 1 x max_seq x vocab_len
            output = self.model(model_input)
            out_tokens = output.argmax(dim=-1).squeeze(0)
            res.append(self.tokenizer.decode(out_tokens))
        return res

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        loglikelihoods = []
        for (string,), dict_args in [req.args for req in requests]:
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string).tolist(),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.model.config.block_size,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                requests=rolling_token_windows,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods


def prepare_results(results: Any, save_filepath: Path, print_results: bool = True) -> None:
    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(results, indent=2, ensure_ascii=False)
    save_filepath.open("w", encoding="utf-8").write(json_result)
