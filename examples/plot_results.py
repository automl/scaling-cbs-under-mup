from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.figure
import pandas as pd
import yaml
from jsonargparse import CLI
from matplotlib import pyplot as plt

from scales.config import PipelineConfig


def collect_values(config: PipelineConfig, result: dict[str, Any], collected_metrics: list[str]) -> dict[str, Any]:
    res = {}
    for metric in collected_metrics:
        metric_path = metric.split(sep=".")
        if metric_path[0] == "result":
            target = result
            for key in metric_path[1:]:
                target = target[key]
        elif metric_path[0] == "pipeline":
            target = config  # type: ignore
            for key in metric_path[1:]:
                target = getattr(target, key)
        else:
            raise ValueError(f"metric must start with either 'pipeline' or 'result' given: {metric}")
        res.update({metric_path[-1]: target})

    return res


def collect_res(configs_dir: Path, results_dir: Path, collected_metrics: list[str]) -> pd.DataFrame:
    config_files = [f for f in configs_dir.iterdir() if f.is_file() and str(f.name).endswith(".yaml")]
    result_folders = [results_dir / file_path.stem for file_path in config_files]
    res_list: list[dict[str, Any]] = []
    for i, config_file in enumerate(config_files):
        config = PipelineConfig.from_path(config_file)
        result_file = result_folders[i] / "result.yaml"
        with result_file.open(encoding="utf-8") as yaml_file:
            result = yaml.safe_load(yaml_file)
        row = collect_values(config, result, collected_metrics)  # type: ignore
        res_list.append(row)
    return pd.DataFrame(res_list)


class Plotter:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def set_style(self) -> None:
        return

    def plot_nparams_valid(self) -> matplotlib.figure.Figure:
        x = self.data["trainable_params"]
        y = self.data["val_loss"]
        fig = plt.figure()
        plt.plot(x, y)
        plt.xlabel("Trainable params")
        plt.ylabel("Valid Loss")
        plt.title("wikitext-103 validation")
        return fig

    @staticmethod
    def save_fig(fig: matplotlib.figure.Figure, out_path: Path) -> None:
        if not out_path.exists() or not out_path.is_file():
            title = fig.axes[0].get_title()
            out_path = out_path / f"{title}.png"
        fig.savefig(str(out_path))


def main(configs_dir: Path | str, results_dir: Path | str, collected_metrics: list[str]) -> None:
    data = collect_res(Path(configs_dir), Path(results_dir), collected_metrics)
    p = Plotter(data)
    fig = p.plot_nparams_valid()
    p.save_fig(fig, Path(results_dir))


if __name__ == "__main__":
    CLI(main)
