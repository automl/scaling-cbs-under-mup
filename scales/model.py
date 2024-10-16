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
from litgpt.model import (
    GPT,
    Block,
    CausalSelfAttention,
    GptNeoxMLP,
    KVCache,
    LLaMAMLP,
    apply_rope,
)
from litgpt.pretrain import reset_parameters
from mup import MuReadout


class file_data_share:
    """This class is mainly used for easy and quick data transfer between different files and methods."""

    layer_wise_max_attn_weight: list = []

    @staticmethod
    def clear_data() -> None:
        file_data_share.layer_wise_max_attn_weight = []


class GPT_Scales(GPT):
    """Overloading of the LitGPT class to use muP.

    Following instructions from https://github.com/microsoft/mup?tab=readme-ov-file#basic-usage

    """

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

        self.val_steps = 0
        self.hs_l1 = [0 for _ in range(config.n_layer + 2)]
        self.hs_l2 = [0 for _ in range(config.n_layer + 2)]

    def update_val_steps(self, steps: int) -> None:
        self.val_steps = steps + 1

    def get_features(self, type: str = "l2") -> list:
        if type == "l2":
            return [(h / self.val_steps) for h in self.hs_l2]
        if type == "l1":
            return [(h / self.val_steps) for h in self.hs_l1]
        raise ValueError("Wrong value for `type`, should be either `l1` or `l2`")

    def clear_features(self) -> None:
        self.hs_l1 = [0 for _ in range(self.config.n_layer + 2)]
        self.hs_l2 = [0 for _ in range(self.config.n_layer + 2)]

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)
        if not self.training:
            self.hs_l2[0] += torch.mean(x**2).item()
            self.hs_l1[0] += torch.mean(torch.abs(x)).item()

        for i, block in enumerate(self.transformer.h):
            x = block(x, cos, sin, mask, input_pos)
            if not self.training:
                self.hs_l2[i + 1] += torch.mean(x**2).item()
                self.hs_l1[i + 1] += torch.mean(torch.abs(x)).item()

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)
        if not self.training:
            self.hs_l2[-1] += torch.mean(logits**2).item()
            self.hs_l1[-1] += torch.mean(torch.abs(logits)).item()

        return logits


class Block_Scales(Block):
    def __init__(self, config: Config, mup_init: bool = False) -> None:
        super().__init__(config)
        self.attn = CausalSelfAttention_Scales(config, mup_init)


class CausalSelfAttention_Scales(CausalSelfAttention):
    def __init__(self, config: Config, mup_init: bool = False) -> None:
        super().__init__(config)
        self.mup_init = mup_init
        if hasattr(config, "apply_qk_norm") and config.apply_qk_norm:
            self.q_norm = config.norm_class(self.config.head_size * self.config.n_head, eps=config.norm_eps)
            self.k_norm = config.norm_class(self.config.head_size * self.config.n_query_groups, eps=config.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        # reshape and apply qk_norm if set
        q = self.q_norm(q.permute(0, 2, 1, 3).reshape(B, T, -1))  # (B, T, nh_q * head_size)
        k = self.k_norm(k.permute(0, 2, 1, 3).reshape(B, T, -1))  # (B, T, nh_k * head_size)

        # reshape back to original shape
        q = q.reshape(B, T, -1, self.config.head_size).permute(0, 2, 1, 3)  # (B, nh_q, T, hs)
        k = k.reshape(B, T, -1, self.config.head_size).permute(0, 2, 1, 3)  # (B, nh_k, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

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
    init_type: Literal["plain", "scaled", "GPT-NeoX", "DeepSeek"] | None = None,
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
    elif init_type == "DeepSeek":
        for mod in model.modules():
            if isinstance(mod, (nn.Embedding, nn.Linear)):
                sigma = 0.006  # TODO: mention source
                mod.reset_parameters = partial(init_weights, mod, std=sigma, mup_init=False)
    else:
        fabric.print("Using standard parametrization")

    if not isinstance(fabric.strategy, FSDPStrategy) and init_type:
        reset_parameters(model)
    else:
        warn(f"Cannot initialize network with current strategy {fabric.strategy}, using standard parametrization")
