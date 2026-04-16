import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

# -----------------------
# Data
# -----------------------
data = {
    "Deep":        {"acc": 0.5425,   "cost": 128.06},
    "Wide":        {"acc": 0.6525,  "cost": 8.19},
    "GROBID":      {"acc": 0.595, "cost": 9.26},
    "LLM-text":    {"acc": 0.6525, "cost": 143.14},
    "LLM-vision": {"acc": 0.67, "cost": 94.3},
    "True":    {"acc": 0.735, "cost": 10.55},
    "SHED":  {"acc": 0.7275,  "cost": 9.13},
}

styles = {
    "Deep":       ("#d62728", "H"),
    "Wide":       ("#1f77b4", "s"),
    "GROBID":     ("#2ca02c", "^"),
    "LLM-text":   ("#9467bd", "D"),
    "LLM-vision": ("#e377c2", "p"),  # bright pink + star
    "SHED": ("#ff7f0e", "P"),
    "True":   ("#17becf", "o"),
}

# -----------------------
# Figure
# -----------------------
# fig, ax = plt.subplots(figsize=(4, 1.8), constrained_layout=True)
fig, ax = plt.subplots(figsize=(2.5, 1.875))

# ax.margins(x=0.03, y=0.05)
ax.margins(x=0.01, y=0.01)

# plt.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.22)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
ax.tick_params(pad=0)
ax.xaxis.labelpad = 0.1
ax.yaxis.labelpad = 0.1

# -----------------------
# Plot
# -----------------------
for k, v in data.items():
    color, marker = styles[k]
    x, y = v["cost"], v["acc"]

    ax.scatter(
        x, y,
        s=110,
        color=color,
        marker=marker,
        edgecolor="black",
        linewidth=0.6,
        zorder=3
    )

    if k == "True":
        ax.text(x * 1.75, y + 0.02, k, fontsize=9, horizontalalignment="right")
    elif k == "SHED":
        ax.text(x * 1.4, y - 0.016, k, fontsize=9, horizontalalignment="left")
    elif k == "LLM-text":
        ax.text(x, y - 0.029, k, fontsize=9, horizontalalignment="center")
    elif k == "LLM-vision":
        ax.text(x, y - 0.029, k, fontsize=9, horizontalalignment="center")
    elif k == "GROBID" or k == "Wide":
        ax.text(x * 1.6, y - 0.02, k, fontsize=9, horizontalalignment="left")
    else:
        ax.text(x * 1.04, y - 0.016, k, fontsize=9)

# -----------------------
# Uniform linear x-axis tuning
# -----------------------

# Tight but not clipping extremes
ax.set_xlim(0, 160)

# Major ticks: human-friendly spacing
# ax.xaxis.set_major_locator(MultipleLocator(20))   # 0, 20, 40, 60, 80, 100
# ax.xaxis.set_minor_locator(MultipleLocator(10))

# Optional: force integer-looking ticks
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

# Y axis nice ticks
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# -----------------------
# Labels
# -----------------------
ax.set_xlabel("Total Cost (USD)", fontsize=10)
ax.set_ylabel("Average Accuracy", fontsize=10)

# -----------------------
# Grid (light paper style)
# -----------------------
ax.grid(True, which="major", linestyle="--", alpha=0.25)
ax.grid(True, which="minor", linestyle=":", alpha=0.15)
# light gray background
ax.set_facecolor("#f9f9f9")

ax.set_ylim(0.52, 0.78)
# 0.5, 0.6, 0.7, 0.8 (one decimal place)
# ensure one decimal place on y-axis
ax.set_yticks(np.arange(0.55, 0.8, 0.1))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.set_xticks(np.arange(0, 151, 30))
# set tick font size
# Make ticks closer to axes
ax.tick_params(axis='both', which='major', pad=0, labelsize=9)
ax.tick_params(axis='both', which='minor', pad=0, labelsize=9)

plt.savefig("sht_ablation_acc_cost.pdf", bbox_inches="tight")