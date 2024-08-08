from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import seaborn as sns
import yaml
from jsonargparse import CLI
from matplotlib import pyplot as plt

from scales.config import PipelineConfig


def collect_values(
    config: PipelineConfig, result: dict[str, Any], info: dict[str, Any], collected_metrics: list[str]
) -> dict[str, Any]:
    res: dict[str, Any] = {}
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
        elif metric_path[0] == "info":
            target = info
            for key in metric_path[1:]:
                if target is not None:
                    target = target.get(key, {})
        else:
            raise ValueError(f"metric must start with either 'pipeline', 'result' or `info` given: {metric}")
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
        info_file = result_folders[i] / "info.yaml"
        with info_file.open(encoding="utf-8") as yaml_file:
            info = yaml.safe_load(yaml_file)
        row = collect_values(config, result, info, collected_metrics)
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
    def create_line_plot(
        ax: matplotlib.axis.Axis,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        """Create a line plot on the given Axes object, with different colors for each value in color_col.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to plot on.
            df (pandas.DataFrame): The dataframe containing the data.
            x_col (str): The column name for the x-axis.
            y_col (str): The column name for the y-axis.
            color_col (str): The column name for determining line colors.
            title (str, optional): The title of the line plot.
            xlabel (str, optional): The label for the x-axis.
            ylabel (str, optional): The label for the y-axis.

        """
        unique_colors = df[color_col].unique()
        palette = sns.color_palette("hsv", len(unique_colors))
        color_dict = dict(zip(unique_colors, palette))

        for color_value in unique_colors:
            subset = df[df[color_col] == color_value]
            ax.plot(subset[x_col], subset[y_col], label=f"{color_col}={color_value}", color=color_dict[color_value])

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    @staticmethod
    def create_subplots_figure(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        subplot_col: str,
        nrows: int,
        ncols: int,
        figsize: tuple[int, int] = (15, 10),
    ) -> matplotlib.figure.Figure:
        """Create a figure with multiple subplots based on the subplot_col.

        Parameters:
            df (pandas.DataFrame): The dataframe containing the data.
            x_col (str): The column name for the x-axis.
            y_col (str): The column name for the y-axis.
            color_col (str): The column name for determining line colors.
            subplot_col (str): The column name for determining subplots.
            nrows (int): Number of rows of subplots.
            ncols (int): Number of columns of subplots.
            figsize (tuple, optional): The size of the figure.

        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()  # Flatten the array of axes for easy iteration

        unique_subplots = df[subplot_col].unique()

        for ax, subplot_value in zip(axes[: len(unique_subplots)], unique_subplots):
            subset = df[df[subplot_col] == subplot_value]
            Plotter.create_line_plot(
                ax, subset, x_col, y_col, color_col, title=f"{subplot_col}={subplot_value}", xlabel=x_col, ylabel=y_col
            )

        # Hide any unused subplots
        for ax in axes[len(unique_subplots) :]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    @staticmethod
    def save_fig(fig: matplotlib.figure.Figure, out_path: Path) -> None:
        # if not out_path.exists() or not out_path.is_file():
        #     title = fig.axes[0].get_title()
        #     out_path = out_path / f"{title}.png"
        fig.savefig(str(out_path))


def main(configs_dir: Path | str, results_dir: Path | str, collected_metrics: list[str]) -> None:
    data = collect_res(Path(configs_dir), Path(results_dir), collected_metrics).fillna(10.0)
    p = Plotter(data)
    fig = p.create_subplots_figure(
        data, x_col="n_head", y_col="val_loss", color_col="d_model", subplot_col="n_layer", ncols=2, nrows=2
    )
    p.save_fig(fig, Path(results_dir) / "mulitline_2.png")


if __name__ == "__main__":
    CLI(main)
