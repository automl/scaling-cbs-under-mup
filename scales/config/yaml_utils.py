from __future__ import annotations

from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import yaml


def function_representer(dumper: yaml.SafeDumper, value: Callable[..., Any]) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!func", {"module": value.__module__, "name": value.__name__})


def function_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Callable[..., Any]:
    mapping_ = loader.construct_mapping(node)
    module = import_module(mapping_["module"], package="scales")
    return getattr(module, mapping_["name"])


def partial_representer(dumper: yaml.SafeDumper, value: partial) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping(  # typing: ignore
        "!partial", {"function": function_representer(dumper, value.func), "kwargs": value.keywords}
    )


def partial_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> partial:
    mapping_ = loader.construct_mapping(node, deep=True)
    func = function_constructor(loader, mapping_["function"])
    kwargs = mapping_["kwargs"]
    return partial(func, **kwargs)


def path_representer(dumper: yaml.SafeDumper, value: Path) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("!path", str(value))


def path_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Path:
    return Path(str(loader.construct_scalar(node)))


def type_representer(dumper: yaml.SafeDumper, value: type) -> yaml.nodes.MappingNode:
    return dumper.represent_mapping("!type", {"module": value.__module__, "name": value.__name__})


def type_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> type:
    mapping_ = loader.construct_mapping(node)
    module = import_module(mapping_["module"], package="scales")
    return getattr(module, mapping_["name"])
