from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

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
    lm_head_bias: bool = False
    _initialized: bool = False

    def __post_init__(self) -> None:
        self.config = Config(
            n_embd=self.d_model,
            n_head=self.n_head,
            n_layer=self.n_layer,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
        )
        super().__setattr__("_initialized", True)
        super().__post_init__()
        self.ignore_fields.append("_initialized")

    def to_dict(self, ignore_defaults: bool = False) -> dict[str, Any]:
        return super().to_dict(ignore_defaults=ignore_defaults) if ignore_defaults else asdict(self.config)

    def __getattr__(self, item: str) -> Any:
        if not self._initialized:
            raise AttributeError(f"'ConfigWrapper' object has no attribute '{item}'")
        if item == "d_model":
            item = "n_embd"
        return getattr(self.config, item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "config":
            super().__setattr__(key, value)
        elif key in ["d_model", "n_head", "n_layer", "block_size", "vocab_size"]:
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
