from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any

import yaml


@dataclass
class BaseConfig:
    """Base class to load and save yaml files for configurations."""

    def __post_init__(self) -> None:
        self.ignore_fields: list[str] = []

    def serialized(self) -> dict[str, Any]:
        # TODO: make the class reconstructable from the yaml file
        # TODO: Add some custom tags to do this
        def serialize(value: Any) -> Any:
            if isinstance(value, partial):
                return {"function": f"{value.func.__module__}.{value.func.__name__}", "kwargs": value.keywords}
            if isinstance(value, FunctionType):
                return f"{value.__module__}.{value.__name__}"
            if isinstance(value, Path):
                return str(value)
            return value

        return {key: serialize(value) for key, value in asdict(self).items() if key not in self.ignore_fields}

    def write_yaml(self, output_dir: Path) -> None:
        ser_dict = self.serialized()
        with (output_dir / f"{type(self).__name__}.yaml").open("w", encoding="utf-8") as yaml_file:
            yaml.dump(ser_dict, yaml_file)

    def load_yaml(self, output_dir: Path) -> dict[str, Any]:
        if (yampl_path := output_dir / f"{type(self).__name__}.yaml").exists():
            with yampl_path.open(encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file)
        else:
            return {}
