import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

# -----------------------
# Data
# -----------------------
data = {
    "Deep":        {"acc": 0.54,   "cost": 97.52},
    "Wide":        {"acc": 0.655,  "cost": 6.65},
    "GROBID":      {"acc": 0.5875, "cost": 7.13},
    "LLM-text":    {"acc": 0.6675, "cost": 109.05},
    "LLM-vision": {"acc": 0.685, "cost": 71.4},
    "True":    {"acc": 0.7375, "cost": 7.23},
    "SHED":  {"acc": 0.735,  "cost": 8.62},
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
fig, ax = plt.subplots(figsize=(4, 1.8), constrained_layout=True)

# -----------------------
# Plot
# -----------------------
for k, v in data.items():
    color, marker = styles[k]
    x, y = v["cost"], v["acc"]

    ax.scatter(
        x, y,
        s=130,
        color=color,
        marker=marker,
        edgecolor="black",
        linewidth=0.6,
        zorder=3
    )

    if k == "True":
        ax.text(x * 1.55, y + 0.016, k, fontsize=9, horizontalalignment="right")
    elif k == "SHED":
        ax.text(x * 1.4, y - 0.016, k, fontsize=9, horizontalalignment="left")
    elif k == "LLM-text":
        ax.text(x, y - 0.04, k, fontsize=9, horizontalalignment="center")
    elif k == "LLM-vision":
        ax.text(x, y - 0.03, k, fontsize=9, horizontalalignment="center")
    elif k == "GROBID" or k == "Wide":
        ax.text(x * 1.6, y - 0.02, k, fontsize=9, horizontalalignment="left")
    else:
        ax.text(x * 1.04, y - 0.016, k, fontsize=9)

# -----------------------
# Uniform linear x-axis tuning
# -----------------------

# Tight but not clipping extremes
ax.set_xlim(0, 120)

# Major ticks: human-friendly spacing
ax.xaxis.set_major_locator(MultipleLocator(20))   # 0, 20, 40, 60, 80, 100
ax.xaxis.set_minor_locator(MultipleLocator(10))

# Optional: force integer-looking ticks
ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

# Y axis nice ticks
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# -----------------------
# Labels
# -----------------------
ax.set_xlabel("Total Cost (USD)", fontsize=11)
ax.set_ylabel("Average Accuracy", fontsize=11)

# -----------------------
# Grid (light paper style)
# -----------------------
ax.grid(True, which="major", linestyle="--", alpha=0.25)
ax.grid(True, which="minor", linestyle=":", alpha=0.15)
# light gray background
ax.set_facecolor("#f9f9f9")

ax.set_ylim(0.50, 0.8)
# 0.5, 0.6, 0.7, 0.8 (one decimal place)
# ensure one decimal place on y-axis
ax.set_yticks(np.arange(0.50, 0.81, 0.1))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.set_xticks(np.arange(0, 121, 30))

plt.savefig("sht_ablation_acc_cost.pdf", bbox_inches="tight")