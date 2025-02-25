import os

import matplotlib.pyplot as plt

# x-values in log space of 2
x_values = [16, 32, 64, 128, 256, 512, 1024]

# Perfect scaling values (dotted line)
perfect_scaling = [8000, 4000, 2000, 1000, 500, 250, 125]

# Actual values (line)
actual_values = [8000, 4000, 2000, 1200, 850, 650, 550]

# 0.75 of perfect scaling
scaled_0_75 = [(0.333333333 * value + value) for value in perfect_scaling]

# Define figure dimensions
text_width_in = 5.5129  # LaTeX \textwidth
nrows = 1
ncols = 1

aspect_ratio = 3 / 4

# Calculate figure height
fig_height_in = text_width_in * aspect_ratio

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(text_width_in, fig_height_in))

plt.plot(
    x_values, perfect_scaling, label=r"$S_\mathrm{ideal}(B, \widetilde{\mathcal{L}})$", linestyle="dotted", color="grey"
)
plt.plot(x_values, actual_values, label=r"$S(B, \widetilde{\mathcal{L}})$", color="red", marker="o")
plt.plot(x_values, scaled_0_75, label="Threshold Shift", linestyle="dashed", color="green")

plt.plot(128, 1200, mec="r", marker="o", color="gold")

# Setting log scale for x-axis
plt.xscale("log", base=2)
plt.yscale("log", base=2)

# Adding labels and title
plt.xlabel("Batch Size")
plt.ylabel("Steps")
plt.title("CBS Illustration")

# Adding a legend
plt.legend(fancybox=False, edgecolor="black")

# Show the plot
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

os.makedirs("figures/", exist_ok=True)

plt.savefig("figures/cbs_illustration.pdf", format="pdf", dpi=600)
plt.show()
