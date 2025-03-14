from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, TypeVar

from litgpt.config import Config

from scales.config.base_config import BaseConfig

T = TypeVar("T", bound="ConfigWrapper")


@dataclass
class ConfigWrapper(BaseConfig):
    """Wrapper around litgpt.Config class tries to imitate a litgpt.Config instance.

    NOTE: to run litgpt.Config.__post_init__ it's necessary to reinitialize the wrapper class

    """

    d_model: int
    n_head: int
    n_layer: int
    block_size: int = 1024
    vocab_size: int = 50257
    bias: bool = True
    apply_qk_norm: bool = False
    parallel_residual: bool = True
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    rotary_percentage: float = 0.25
    norm_eps: float = 1e-5
    lm_head_bias: bool = False
    attn_residual_weight: float = 0.5
    rec_mlp_weight: float = 0.5
    _initialized: bool = False

    def __post_init__(self) -> None:
        self.config = Config(
            n_embd=self.d_model,
            n_head=self.n_head,
            n_layer=self.n_layer,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            norm_class_name=self.norm_class_name,
            rotary_percentage=self.rotary_percentage,
            parallel_residual=self.parallel_residual,
            norm_eps=self.norm_eps,
            bias=self.bias,
            lm_head_bias=self.lm_head_bias,
        )
        super().__setattr__("_initialized", True)
        super().__post_init__()
        self.ignore_fields.append("_initialized")

    def to_dict(self, ignore_defaults: bool = False) -> dict[str, Any]:
        # TODO: Add a way to priint all config attributes as well
        return super().to_dict(ignore_defaults=ignore_defaults)

    def __getattr__(self, item: str) -> Any:
        if not self._initialized:
            raise AttributeError(f"'ConfigWrapper' object has no attribute '{item}'")
        if item == "d_model":
            item = "n_embd"
        return getattr(self.config, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "config":
            super().__setattr__(key, value)
        # Perhaps this check should be removed
        elif key in [field.name for field in fields(ConfigWrapper)] and key != "_initialized":
            super().__setattr__(key, value)
            if self._initialized:
                if key == "d_model":
                    key = "n_embd"
                setattr(self.config, key, value)
        else:
            if self._initialized:
                setattr(self.config, key, value)
            else:
                super().__setattr__(key, value)

    @classmethod
    def from_config(cls: type[T], config: Config) -> T:
        param_names = [field.name for field in fields(cls)]
        params = []
        for name in param_names:
            attr_name = name
            if name == "d_model":
                attr_name = "n_embd"
            if hasattr(config, attr_name):
                params.append((name, getattr(config, attr_name)))
        return cls(**dict(params))


if __name__ == "__main__":
    w = ConfigWrapper(32, 2, 3)
    print(asdict(w))
    print(asdict(w.config))
    print(w.block_size)
    w.block_size = 512
    print(w.block_size)
    print(w.config.block_size)
    print(asdict(w.config))
