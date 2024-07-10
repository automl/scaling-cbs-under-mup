import math
from typing import Optional

import torch
import torch.nn as nn
from litgpt.config import Config
from litgpt.model import GPT, Block, CausalSelfAttention
from mup import MuReadout


class file_data_share:
    """This class is mainly used for easy and quick data transfer between different files and methods."""

    layer_wise_max_attn_weight: list = []

    @staticmethod
    def clear_data() -> None:
        file_data_share.layer_wise_max_attn_weight = []


class GPT_Scales(GPT):
    def __init__(self, config: Config, mup: bool = False) -> None:
        super().__init__(config)
        if mup:
            self.lm_head = MuReadout(
                config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias, readout_zero_init=True
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(Block_Scales(config, mup) for _ in range(config.n_layer)),
                "ln_f": config.norm_class(config.n_embd, eps=config.norm_eps),
            }
        )


class Block_Scales(Block):
    def __init__(self, config: Config, mup: bool = False) -> None:
        super().__init__(config)
        self.attn = CausalSelfAttention_Scales(config, mup)


class CausalSelfAttention_Scales(CausalSelfAttention):
    def __init__(self, config: Config, mup: bool = False) -> None:
        super().__init__(config)
        self.mup = mup

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / self.config.head_size if self.mup else 1.0 / math.sqrt(self.config.head_size)

        L, S = q.size(-2), k.size(-2)

        if self.mup:
            scale_factor = 1 / (q.size(-1)) if scale is None else scale
        else:
            scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

        attn_bias = torch.zeros(L, S, dtype=q.dtype)
        if mask is None:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_bias = attn_bias.to(attn_weight.device)
        attn_weight += attn_bias

        file_data_share.layer_wise_max_attn_weight.append(torch.max(attn_weight).item())

        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = attn_weight @ v

        return y.transpose(1, 2)
