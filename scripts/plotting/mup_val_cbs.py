import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from cycler import cycler
from litgpt.config import Config

from scales.utils import count_trainable_parameters_chinchilla

file_path_mup_wb32_wt32 = [
    "../output/final_results/seed444/mup_wb32_wt32_tpp20",
    "../output/final_results/seed42/mup_wb32_wt32_tpp20",
    "../output/final_results/seed2/mup_wb32_wt32_tpp20",
]
file_path_mup_wb32_wt96 = [
    "../output/final_results/seed444/mup_wb32_wt96_tpp20",
    "../output/final_results/seed42/mup_wb32_wt96_tpp20",
    "../output/final_results/seed2/mup_wb32_wt96_tpp20",
]
file_path_mup_wb32_wt288 = [
    "../output/final_results/seed444/mup_wb32_wt288_tpp20",
    "../output/final_results/seed42/mup_wb32_wt288_tpp20",
    "../output/final_results/seed2/mup_wb32_wt288_tpp20",
]
file_path_mup_wb32_wt768 = [
    "../output/final_results/seed444/mup_wb32_wt768_tpp20",
    "../output/final_results/seed42/mup_wb32_wt768_tpp20",
    "../output/final_results/seed2/mup_wb32_wt768_tpp20",
]

# Font size settings
font_size_base = 10  # Base font size, adjust to match your LaTeX document
matplotlib.rcParams.update(
    {
        "font.size": font_size_base,  # General font size
        "axes.titlesize": font_size_base,  # Title font size
        "axes.labelsize": font_size_base,  # Axis labels
        "xtick.labelsize": font_size_base - 1,  # X tick labels
        "ytick.labelsize": font_size_base - 1,  # Y tick labels
        "legend.fontsize": font_size_base - 1,  # Legend text
        "xtick.direction": "in",  # X ticks pointing inside
        "ytick.direction": "in",  # Y ticks pointing inside
        "axes.prop_cycle": cycler(
            color=["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
        ),
        "font.family": "serif",
        "text.usetex": True,
    }
)
colors_from_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors_for_plots = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#a65628", "#ff7f00", "#ffff33", "#f781bf"]

config_w32 = Config(block_size=1024, n_layer=8, n_head=4, vocab_size=50257, bias=True, n_embd=32)
config_w96 = Config(block_size=1024, n_layer=8, n_head=4, vocab_size=50257, bias=True, n_embd=96)
config_w288 = Config(block_size=1024, n_layer=8, n_head=4, vocab_size=50257, bias=True, n_embd=288)
config_w768 = Config(block_size=1024, n_layer=8, n_head=4, vocab_size=50257, bias=True, n_embd=768)


def get_bs_and_val_loss(file_path: str | list[str], config: Config, tokens_per_param: int = 20):
    if isinstance(file_path, str):
        file_paths = [file_path]  # Wrap single path in a list for uniform processing
    else:
        file_paths = file_path

    batch_sizes = []
    validation_losses = {}
    steps_per_batch = []

    for path in file_paths:
        for folder_name in os.listdir(path):
            if folder_name.startswith("bs"):
                bs_number = int(folder_name[2:])
                if bs_number not in batch_sizes:
                    batch_sizes.append(bs_number)

    batch_sizes.sort()

    # Initialize validation_losses as a dictionary to store lists for each batch size
    for bs in batch_sizes:
        validation_losses[bs] = []

    for path in file_paths:
        for bs_number in batch_sizes:
            folder_name = f"bs{bs_number}"  # Reconstruct the folder name
            folder_path = os.path.join(path, folder_name)

            for file_name in os.listdir(folder_path):
                if "tb_logs.csv" in file_name:
                    csv_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(csv_path)

                    np_val_loss = df["Validation Loss"].to_numpy()
                    np_steps = df["step"].to_numpy()
                    np_tokens_per_step = df["Tokens-Per-Step"].to_numpy()

                    if tokens_per_param:
                        max_tokens_per_config = count_trainable_parameters_chinchilla(config=config) * tokens_per_param

                        valid_indices = np.where(np_tokens_per_step <= max_tokens_per_config)[0][-1]
                        validation_losses[bs_number].append(np_val_loss[valid_indices])

                    # Record the last validation loss and steps
                    if len(steps_per_batch) < len(batch_sizes):
                        steps_per_batch.append(np_steps[-1] - 1)

    return batch_sizes, validation_losses, steps_per_batch


def get_mean_std_losses(val_losses):
    means = []
    stds = []
    for _, losses in val_losses.items():
        means.append(np.mean(losses))
        stds.append(np.std(losses))

    return means, stds


def get_mean_std_delta_losses(val_losses, batch_size):
    delta_losses = {}

    for batch_size, losses in val_losses.items():
        best_loss = min(losses)
        delta_losses[batch_size] = [loss - best_loss for loss in losses]

    num_indices = len(next(iter(val_losses.values())))

    best_loss = []
    for i in range(num_indices):
        values_at_i = [value[i] for value in val_losses.values()]
        best_loss.append(min(values_at_i))

    for key, values in val_losses.items():
        delta_losses[key] = [values[i] - best_loss[i] for i in range(len(values))]

    stats = {}

    for key, values in delta_losses.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats[key] = {"mean": mean_val, "std": std_val}

    return stats


batch_sizes_mup32, validation_losses_mup32, steps_per_batch_mup32 = get_bs_and_val_loss(
    file_path_mup_wb32_wt32, config_w32, tokens_per_param=20
)
batch_sizes_mup96, validation_losses_mup96, steps_per_batch_mup96 = get_bs_and_val_loss(
    file_path_mup_wb32_wt96, config_w96, tokens_per_param=20
)
batch_sizes_mup288, validation_losses_mup288, steps_per_batch_mup288 = get_bs_and_val_loss(
    file_path_mup_wb32_wt288, config_w288, tokens_per_param=20
)
batch_sizes_mup768, validation_losses_mup768, steps_per_batch_mup768 = get_bs_and_val_loss(
    file_path_mup_wb32_wt768, config_w768, tokens_per_param=20
)

stats_mup32 = get_mean_std_delta_losses(validation_losses_mup32, batch_sizes_mup32)
mean_losses_mup32, std_losses_mup32 = get_mean_std_losses(validation_losses_mup32)
stats_mup96 = get_mean_std_delta_losses(validation_losses_mup96, batch_sizes_mup96)
mean_losses_mup96, std_losses_mup96 = get_mean_std_losses(validation_losses_mup96)
stats_mup288 = get_mean_std_delta_losses(validation_losses_mup288, batch_sizes_mup288)
mean_losses_mup288, std_losses_mup288 = get_mean_std_losses(validation_losses_mup288)
stats_mup768 = get_mean_std_delta_losses(validation_losses_mup768, batch_sizes_mup768)
mean_losses_mup768, std_losses_mup768 = get_mean_std_losses(validation_losses_mup768)


keys_mup32 = list(stats_mup32.keys())
means_mup32 = [stats_mup32[key]["mean"] for key in keys_mup32]
stds_mup32 = [stats_mup32[key]["std"] for key in keys_mup32]


keys_mup96 = list(stats_mup96.keys())
means_mup96 = [stats_mup96[key]["mean"] for key in keys_mup96]
stds_mup96 = [stats_mup96[key]["std"] for key in keys_mup96]

keys_mup288 = list(stats_mup288.keys())
means_mup288 = [stats_mup288[key]["mean"] for key in keys_mup288]
stds_mup288 = [stats_mup288[key]["std"] for key in keys_mup288]

keys_mup768 = list(stats_mup768.keys())
means_mup768 = [stats_mup768[key]["mean"] for key in keys_mup768]
stds_mup768 = [stats_mup768[key]["std"] for key in keys_mup768]

# Data for each model size
models = {
    "$\mu$P 1.7M": (batch_sizes_mup32, mean_losses_mup32, std_losses_mup32, means_mup32, stds_mup32),
    "$\mu$P 5.6M": (batch_sizes_mup96, mean_losses_mup96, std_losses_mup96, means_mup96, stds_mup96),
    "$\mu$P 20.5M": (batch_sizes_mup288, mean_losses_mup288, std_losses_mup288, means_mup288, stds_mup288),
    "$\mu$P 81.4M": (batch_sizes_mup768, mean_losses_mup768, std_losses_mup768, means_mup768, stds_mup768),
}

colors = {model: color for model, color in zip(models, colors_for_plots[: len(models)])}

text_width_in = 5.5129  # LaTeX \textwidth
nrows = 1
ncols = 2

aspect_ratio = 2 / 3

fig_height_in = text_width_in * aspect_ratio

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(text_width_in, fig_height_in))

ax1, ax2 = axes

# First plot: Validation Curves vs. Batch Sizes
for model_name, (batch_sizes, val_losses_mean, val_losses_std, delta_losses_mean, delta_losses_std) in models.items():
    ax1.plot(batch_sizes, val_losses_mean, label=model_name, marker="o", color=colors[model_name])
    ax1.errorbar(
        batch_sizes,
        val_losses_mean,
        yerr=val_losses_std,
        fmt="o",
        color=colors[model_name],
        capsize=5,
        elinewidth=2,
    )

ax1.set_xscale("log", base=2)
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Validation Loss")
ax1.legend(fancybox=False, edgecolor="black")
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

for model_name, (batch_sizes, val_losses_mean, val_losses_std, delta_losses_mean, delta_losses_std) in models.items():
    ax2.plot(batch_sizes, delta_losses_mean, label=model_name, marker="o", color=colors[model_name])
    ax2.errorbar(
        batch_sizes,
        delta_losses_mean,
        yerr=delta_losses_std,
        fmt="o",
        color=colors[model_name],
        capsize=5,
        elinewidth=2,
    )

ax2.set_xscale("log", base=2)
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Delta Validation Loss ")

ax2.set_ylim(-0.02, 0.7)
ax2.legend(fancybox=False, edgecolor="black")
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plots
plt.tight_layout()
os.makedirs("figures/", exist_ok=True)

plt.savefig("figures/validation_and_sensitivity_curve_mup.pdf")
plt.show()


def get_best_validation_loss(file_path: str | list[str]) -> np.ndarray:
    # If the input is a list of paths, process each path individually and return an array of best losses
    if isinstance(file_path, list):
        best_losses = []
        for path in file_path:
            best_loss = get_best_loss_in_folder(path)
            best_losses.append(best_loss)
        return np.array(best_losses)

    # If the input is a single path, process that folder directly
    return np.array([get_best_loss_in_folder(file_path)])


def get_best_loss_in_folder(folder_path: str) -> float:
    best_loss = np.inf

    # Iterate through each folder in the directory
    for folder_name in os.listdir(folder_path):
        if folder_name.startswith("bs"):
            folder_path_full = os.path.join(folder_path, folder_name)

            # Define the path to the results.yaml file
            yaml_file_path = os.path.join(folder_path_full, "result.yaml")

            # Check if the file exists before loading
            if os.path.isfile(yaml_file_path):
                # Load the YAML file
                with open(yaml_file_path, "r") as file:
                    data = yaml.safe_load(file)

                # Extract the validation loss value
                val_loss = data.get("val_loss")
                if val_loss is not None and val_loss < best_loss:
                    best_loss = val_loss

    return best_loss


def get_losses_percentage(val_losses: list, percent: float = 7.5):
    percent_val_losses = []
    for val in val_losses:
        percent_val_losses.append(val * (percent / 100) + val)

    return percent_val_losses


def get_cbs(file_path: str | list[str], val_losses: list):
    if isinstance(file_path, str):
        file_paths = [file_path]
    else:
        file_paths = file_path

    batch_sizes = []
    steps = {}

    for path in file_paths:
        for folder_name in os.listdir(path):
            if folder_name.startswith("bs"):
                bs_number = int(folder_name[2:])
                if bs_number not in batch_sizes:
                    batch_sizes.append(bs_number)

    batch_sizes.sort()

    for bs in batch_sizes:
        steps[bs] = []

    for i, path in enumerate(file_paths):
        for bs_number in batch_sizes:
            folder_name = f"bs{bs_number}"
            folder_path = os.path.join(path, folder_name)

            for file_name in os.listdir(folder_path):
                if "tb_logs.csv" in file_name:
                    csv_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(csv_path)

                    np_val_loss = df["Validation Loss"].to_numpy()
                    np_steps = df["step"].to_numpy()

                    indices = np.where(np_val_loss <= val_losses[i])[0]
                    index = indices[0] if indices.size > 0 else None

                    if index is not None:
                        steps[bs_number].append(np_steps[index])
                        step_found = True
                        break

    steps_mean = []
    steps_std = []
    for batch_size, step in steps.items():
        if len(step) != 0:
            steps_mean.append(np.mean(step))
            steps_std.append(np.std(step))

    perfect_scaling_steps = [(steps_mean[0] * 0.0 + steps_mean[0]) / (2**i) for i in range(len(batch_sizes))]
    scale = [perfect_scaling_steps[i] / steps_mean[i] for i in range(len(steps_mean))]
    for i in range(len(scale) - 1, -1, -1):
        if scale[i] > 0.75:
            last_index_above_0_75 = i
            break

    critical_batch_size = batch_sizes[last_index_above_0_75]

    return batch_sizes, steps_mean, steps_std, perfect_scaling_steps, critical_batch_size


# Helper function to find the index of a critical batch size in a numpy array
def get_index(arr, value):
    index_array = np.where(arr == value)[0]
    if index_array.size > 0:
        return index_array[0]
    else:
        return None


# Example function calls for `get_cbs` (assuming these are defined)
batch_sizes_mup32, steps_mean_mup32, steps_std_mup32, perfect_scaling_steps_mup32, critical_batch_size_mup32 = get_cbs(
    file_path_mup_wb32_wt32, get_losses_percentage(get_best_validation_loss(file_path_mup_wb32_wt32), 7.5)
)
batch_sizes_mup96, steps_mean_mup96, steps_std_mup96, perfect_scaling_steps_mup96, critical_batch_size_mup96 = get_cbs(
    file_path_mup_wb32_wt96, get_losses_percentage(get_best_validation_loss(file_path_mup_wb32_wt96), 7.5)
)
batch_sizes_mup288, steps_mean_mup288, steps_std_mup288, perfect_scaling_steps_mup288, critical_batch_size_mup288 = (
    get_cbs(file_path_mup_wb32_wt288, get_losses_percentage(get_best_validation_loss(file_path_mup_wb32_wt288), 7.5))
)
batch_sizes_mup768, steps_mean_mup768, steps_std_mup768, perfect_scaling_steps_mup768, critical_batch_size_mup768 = (
    get_cbs(file_path_mup_wb32_wt768, get_losses_percentage(get_best_validation_loss(file_path_mup_wb32_wt768), 7.5))
)

models = {
    "$\mu$P 1.7M": (
        batch_sizes_mup32,
        steps_mean_mup32,
        steps_std_mup32,
        perfect_scaling_steps_mup32,
        critical_batch_size_mup32,
    ),
    "$\mu$P 5.6M": (
        batch_sizes_mup96,
        steps_mean_mup96,
        steps_std_mup96,
        perfect_scaling_steps_mup96,
        critical_batch_size_mup96,
    ),
    "$\mu$P 20.5M": (
        batch_sizes_mup288,
        steps_mean_mup288,
        steps_std_mup288,
        perfect_scaling_steps_mup288,
        critical_batch_size_mup288,
    ),
    "$\mu$P 81.4M": (
        batch_sizes_mup768,
        steps_mean_mup768,
        steps_std_mup768,
        perfect_scaling_steps_mup768,
        critical_batch_size_mup768,
    ),
}

colors = {model: color for model, color in zip(models, colors_for_plots[: len(models)])}

light_gray = "#d3d3d3"


# Define figure dimensions
text_width_in = 5.5129  # LaTeX \textwidth
nrows = 2
ncols = 2

aspect_ratio = 3 / 4

# Calculate figure height
fig_height_in = text_width_in * aspect_ratio

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(text_width_in, fig_height_in))
for (model_name, (batch_sizes, step_scaling, step_std, perfect_scaling, critical_batch_size)), ax in zip(
    models.items(), axes.flat
):
    crit_bs_index = get_index(np.array(batch_sizes), critical_batch_size)
    ax.plot(
        batch_sizes[: len(step_scaling)],
        perfect_scaling[: len(step_scaling)],
        linestyle=":",
        color=light_gray,
        zorder=1,
    )
    ax.plot(
        batch_sizes[: len(step_scaling)],
        step_scaling,
        label="Step Scaling",
        color=colors[model_name],
        marker="o",
        zorder=2,
    )
    # Plot error bars
    ax.errorbar(
        batch_sizes[: len(step_scaling)],
        step_scaling,
        yerr=step_std,
        fmt="o",
        color=colors[model_name],
        capsize=5,
        elinewidth=2,
        zorder=3,
    )
    if crit_bs_index is not None:
        ax.plot(
            batch_sizes[crit_bs_index],
            step_scaling[crit_bs_index],
            mec=colors[model_name],
            marker="o",
            color="gold",
            zorder=4,
        )

    # Set the axis scale and labels with correct font sizes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Steps")
    ax.set_title(f"{model_name}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax.tick_params(axis="both", which="major")

plt.tight_layout()
os.makedirs("figures/", exist_ok=True)

plt.savefig("figures/cbs_mup.pdf", format="pdf", bbox_inches="tight", dpi=600)
plt.show()
