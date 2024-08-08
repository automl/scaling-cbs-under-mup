import math
from functools import partial
from typing import Literal, Optional
from warnings import warn

import lightning as L
import mup
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from litgpt.config import Config
from litgpt.model import GPT, Block, CausalSelfAttention, GptNeoxMLP, LLaMAMLP
from litgpt.pretrain import reset_parameters
from mup import MuReadout


class file_data_share:
    """This class is mainly used for easy and quick data transfer between different files and methods."""

    layer_wise_max_attn_weight: list = []

    @staticmethod
    def clear_data() -> None:
        file_data_share.layer_wise_max_attn_weight = []


class GPT_Scales(GPT):
    def __init__(self, config: Config, mup_init: bool = False) -> None:
        super().__init__(config)
        if mup_init:
            self.lm_head = MuReadout(
                config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias, readout_zero_init=True
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(Block_Scales(config, mup_init) for _ in range(config.n_layer)),
                "ln_f": config.norm_class(config.n_embd, eps=config.norm_eps),
            }
        )


class Block_Scales(Block):
    def __init__(self, config: Config, mup_init: bool = False) -> None:
        super().__init__(config)
        self.attn = CausalSelfAttention_Scales(config, mup_init)


class CausalSelfAttention_Scales(CausalSelfAttention):
    def __init__(self, config: Config, mup_init: bool = False) -> None:
        super().__init__(config)
        self.mup_init = mup_init

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / self.config.head_size if self.mup_init else 1.0 / math.sqrt(self.config.head_size)

        L, S = q.size(-2), k.size(-2)

        if self.mup_init:
            scale_factor = 1 / (q.size(-1)) if scale is None else scale
        else:
            scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

        # a bit more memory efficient version of causal self attention
        # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L67
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_bias = torch.ones(L, S, dtype=q.dtype, device=q.device).tril(diagonal=0).requires_grad_(False)
        attn_weight = attn_weight.masked_fill_(attn_bias == 0, float("-inf"))
        del attn_bias

        file_data_share.layer_wise_max_attn_weight.append(torch.max(attn_weight).detach().item())

        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = attn_weight @ v

        return y.transpose(1, 2)


def initialize_weights(
    fabric: L.Fabric,
    model: GPT_Scales,
    mup_base_scales: dict[str, int] | int | None = None,
    init_type: Literal["plain", "scaled", "GPT-NeoX"] | None = None,
) -> None:
    def init_weights(
        module: nn.Module,
        std: float,
        mup_init: bool,
    ) -> None:
        if mup_init:
            mup.normal_(module.weight, mean=0.0, std=std)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    if init_type == "plain" or init_type == "scaled" or init_type == "GPT-NeoX":
        # "scaled" and "plain" Weight initialization (https://arxiv.org/abs/2312.16903).
        if isinstance(mup_base_scales, int):
            d_model = mup_base_scales
        elif isinstance(mup_base_scales, dict):
            d_model = mup_base_scales["d_model"]
        else:
            d_model = model.config.n_embd
        sigma = math.sqrt(2.0 / (5 * d_model))

        for mod in model.modules():
            if isinstance(mod, (nn.Embedding, nn.Linear)):
                mod.reset_parameters = partial(init_weights, mod, std=sigma, mup_init=mup_base_scales is not None)

        # need a separate loop because `mod.proj` below is a `nn.Linear` too
        if init_type == "scaled":
            for mod in model.modules():
                if isinstance(mod, (LLaMAMLP, CausalSelfAttention_Scales, GptNeoxMLP)):
                    mod.proj.reset_parameters = partial(
                        init_weights,
                        mod.proj,
                        std=(sigma / math.sqrt(model.config.n_layer * 2)),
                        mup_init=mup_base_scales is not None,
                    )
        elif init_type == "GPT-NeoX":
            # GPT-NeoX-20B weight initialization (https://arxiv.org/abs/2204.06745).
            for mod in model.modules():
                if isinstance(mod, (LLaMAMLP, CausalSelfAttention_Scales, GptNeoxMLP)):
                    mod.proj.reset_parameters = partial(
                        init_weights,
                        mod.proj,
                        std=(1 / math.sqrt(d_model) / model.config.n_layer),
                        mup_init=mup_base_scales is not None,
                    )

        if not isinstance(fabric.strategy, FSDPStrategy):
            reset_parameters(model)

    else:
        warn(f"The init_type {init_type} is not supported.")
