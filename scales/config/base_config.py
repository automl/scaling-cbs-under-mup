from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, TypeVar

import yaml

from scales.config.yaml_utils import (
    function_constructor,
    function_representer,
    partial_constructor,
    partial_representer,
    path_constructor,
    path_representer,
    type_constructor,
    type_representer,
)
from scales.lr_utils import LRScheduler


def get_field_default(field: dataclasses.Field) -> Any:
    # A horrible way of getting the default values
    if not isinstance(field.default_factory, type(dataclasses.MISSING)):
        return field.default_factory()  # type: ignore
    return field.default


T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class to load and save yaml files for configurations."""

    def __post_init__(self) -> None:
        self.ignore_fields: list[str] = []

    def to_dict(self, ignore_defaults: bool = False) -> dict[str, Any]:
        dict_ = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)

            default = get_field_default(field)

            # if not in ignored fields and (ignore_defaults --> inequality_check)
            if key not in self.ignore_fields and (default != value or not ignore_defaults):
                dict_[key] = value
                # To handle the Edge cases of having dataclasses as fields
                if issubclass(type(value), BaseConfig):
                    dict_[key] = value.to_dict(ignore_defaults=ignore_defaults)
                elif is_dataclass(value):
                    dict_[key] = asdict(value)
        return dict_

    def write_yaml(self, output_dir: Path, ignore_defaults: bool = True, name: str | None = None) -> None:
        config = self.to_dict(ignore_defaults=ignore_defaults)
        name = name if name is not None else type(self).__name__
        yaml_path = output_dir if output_dir.suffix == ".yaml" else output_dir / f"{name}.yaml"
        print(f"Saving Configration at {str(yaml_path)}")
        with yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.dump(config, yaml_file, Dumper=self.yaml_dumper())

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
        dumper.add_multi_representer(type(LRScheduler), type_representer)
        return dumper

    @classmethod
    def load_yaml(cls: type[T], output_dir: Path) -> dict[str, Any]:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        yaml_path = output_dir if output_dir.suffix == ".yaml" else output_dir / f"{cls.__name__}.yaml"
        if yaml_path.exists():
            with yaml_path.open(encoding="utf-8") as yaml_file:
                loader: type[yaml.SafeLoader] = cls.yaml_loader()
                return yaml.load(yaml_file, Loader=loader)  # noqa S506
        else:
            return {}

    @classmethod
    def from_yaml(cls: type[T], yaml_config: dict[str, Any], yaml_hook: Callable | None = None) -> T:
        return cls(**yaml_config)

    @classmethod
    def from_path(cls: type[T], path: Path, yaml_hook: Callable | None = None) -> T:
        yaml_config = cls.load_yaml(output_dir=path)
        return cls.from_yaml(yaml_config, yaml_hook)
