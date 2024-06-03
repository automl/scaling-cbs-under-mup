from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any, Generic, TypeVar

import yaml
from yaml_utils import (
    function_constructor,
    function_representer,
    partial_constructor,
    partial_representer,
    path_constructor,
    path_representer,
    type_constructor,
    type_representer,
)

from scales.lr_utils import BaseLR

T = TypeVar("T")


@dataclass
class BaseConfig(Generic[T]):
    """Base class to load and save yaml files for configurations."""

    def __post_init__(self) -> None:
        self.ignore_fields: list[str] = []

    def serialized(self) -> dict[str, Any]:
        # TODO: make the class reconstructable from the yaml file
        # TODO: Add some custom tags to do this
        # def serialize(value: Any) -> Any:
        #     if isinstance(value, partial):
        #         return {"function": f"{value.func.__module__}.{value.func.__name__}", "kwargs": value.keywords}
        #     if isinstance(value, FunctionType):
        #         return f"!fun {value.__module__}.{value.__name__}"
        #     if isinstance(value, Path):
        #         return str(value)
        #     if isclass(value):
        #         return f"!class {value.__module__}.{value.__name__}"
        #     return value

        return {key: value for key, value in asdict(self).items() if key not in self.ignore_fields}

    def write_yaml(self, output_dir: Path) -> None:
        ser_dict = self.serialized()
        yaml_path = output_dir / f"{type(self).__name__}.yaml"
        print(f"Saving Configration at {str(yaml_path)}")
        with yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.dump(ser_dict, yaml_file, Dumper=self.yaml_dumper())

    @staticmethod
    def yaml_loader() -> type[yaml.SafeLoader]:
        loader = yaml.SafeLoader
        loader.add_constructor("!func", function_constructor)
        loader.add_constructor("!partial", partial_constructor)
        loader.add_constructor("!path", path_constructor)
        loader.add_constructor("!type", type_constructor)
        return loader

    @staticmethod
    def yaml_dumper() -> type[yaml.SafeDumper]:
        dumper = yaml.SafeDumper
        dumper.add_representer(FunctionType, function_representer)
        dumper.add_representer(partial, partial_representer)
        dumper.add_multi_representer(Path, path_representer)
        dumper.add_multi_representer(type(BaseLR), type_representer)
        return dumper

    @classmethod
    def load_yaml(cls, output_dir: Path) -> dict[str, Any]:
        yaml_path = output_dir if output_dir.suffix == ".yaml" else output_dir / f"{cls.__name__}.yaml"
        if yaml_path.exists():
            with yaml_path.open(encoding="utf-8") as yaml_file:
                loader: type[yaml.SafeLoader] = cls.yaml_loader()
                return yaml.load(yaml_file, Loader=loader)  # noqa S506
        else:
            return {}

    @classmethod
    def from_yaml(cls, yaml_config: dict[str, Any]) -> BaseConfig:
        return cls(**yaml_config)

    @classmethod
    def from_path(cls, path: Path) -> BaseConfig:
        yaml_config = cls.load_yaml(output_dir=path)
        return cls.from_yaml(yaml_config)
