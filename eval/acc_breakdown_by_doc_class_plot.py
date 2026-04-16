import matplotlib.pyplot as plt
import numpy as np

# x-axis
p_vals = np.array([5, 10, 15, 20, 30, 40])

# Doc classes and colors
doc_classes = [
    "well_formatted",
    "loosely_formatted",
    # "depth_aligned",
    # "local_first",
    # "global_first"
]

colors = {
    "well_formatted": "red",
    "loosely_formatted": "green",
    "depth_aligned": "orange",
    "local_first": "blue",
    "global_first": "purple"
}

# Helper to convert None → np.nan
def clean(arr):
    return np.array([np.nan if v is None else v for v in arr], dtype=float)

# -----------------------
# DATA (TRUE / FALSE)
# -----------------------

data = {
    "Civic": {
        "well_formatted": {
            "TRUE":  clean([None, None, None, None, None, None]),
            "FALSE": clean([67.92, 67.32, 67.38, 70.26, 71.42, 71.92]),
        },
        "loosely_formatted": {
            "TRUE":  clean([63.9, 63.07, 63.2, 64.41, 66.35, 67.77]),
            "FALSE": clean([72.38, 72.05, 72.03, 76.75, 77.05, 76.53]),
        },
        "depth_aligned": {
            "TRUE":  clean([None]*6),
            "FALSE": clean([67.92, 67.32, 67.38, 70.26, 71.42, 71.92]),
        },
        "local_first": {
            "TRUE":  clean([63.9, 63.07, 63.2, 64.41, 66.35, 67.77]),
            "FALSE": clean([72.38, 72.05, 72.03, 76.75, 77.05, 76.53]),
        },
        "global_first": {
            "TRUE":  clean([63.9, 63.07, 63.2, 64.41, 66.35, 67.77]),
            "FALSE": clean([72.38, 72.05, 72.03, 76.75, 77.05, 76.53]),
        },
    },

    "Contract": {
        "well_formatted": {
            "TRUE":  clean([62.31, 67.36, 72.4, 78.34, 81.9, 82.49]),
            "FALSE": clean([67.17, 74.1, 78.01, 80.12, 84.34, 84.64]),
        },
        "loosely_formatted": {
            "TRUE":  clean([65.3, 69.2, 73.72, 77.82, 82.34, 82.55]),
            "FALSE": clean([63.19, 74.73, 79.12, 82.97, 85.16, 86.26]),
        },
        "depth_aligned": {
            "TRUE":  clean([50, 40, 65, 70, 75, 75]),
            "FALSE": clean([65.18, 71.65, 75.5, 79.51, 83.36, 83.82]),
        },
        "local_first": {
            "TRUE":  clean([65.3, 69.2, 73.72, 77.82, 82.34, 82.55]),
            "FALSE": clean([63.19, 74.73, 79.12, 82.97, 85.16, 86.26]),
        },
        "global_first": {
            "TRUE":  clean([65.3, 69.2, 73.72, 77.82, 82.34, 82.55]),
            "FALSE": clean([63.19, 74.73, 79.12, 82.97, 85.16, 86.26]),
        },
    },

    "Qasper": {
        "well_formatted": {
            "TRUE":  clean([53.78, 64.05, 65.86, 70.19, 67.98, 71.0]),
            "FALSE": clean([57.04, 64.43, 66.88, 68.04, 70.24, 69.83]),
        },
        "loosely_formatted": {
            "TRUE":  clean([55.56, 63.76, 65.89, 68.12, 69.03, 69.88]),
            "FALSE": clean([57.86, 65.58, 68.25, 69.41, 71.21, 70.56]),
        },
        "depth_aligned": {
            "TRUE":  clean([None]*6),
            "FALSE": clean([56.29, 64.34, 66.64, 68.53, 69.73, 70.09]),
        },
        "local_first": {
            "TRUE":  clean([55.59, 63.7, 65.93, 68.15, 69.06, 69.97]),
            "FALSE": clean([57.81, 65.72, 68.19, 69.35, 71.17, 70.37]),
        },
        "global_first": {
            "TRUE":  clean([55.59, 63.7, 65.93, 68.15, 69.06, 69.97]),
            "FALSE": clean([57.81, 65.72, 68.19, 69.35, 71.17, 70.37]),
        },
    },

    "Finance": {
        "well_formatted": {
            "TRUE":  clean([50, 50, 50, 50, 50, 50]),
            "FALSE": clean([41.84, 40.69, 42.53, 44.6, 43.68, 45.75]),
        },
        "loosely_formatted": {
            "TRUE":  clean([38.89, 41.67, 44.44, 52.78, 50, 47.22]),
            "FALSE": clean([42.22, 40.74, 42.47, 43.95, 43.21, 45.68]),
        },
        "depth_aligned": {
            "TRUE":  clean([None]*6),
            "FALSE": clean([41.95, 40.82, 42.63, 44.67, 43.76, 45.8]),
        },
        "local_first": {
            "TRUE":  clean([38.89, 41.67, 44.44, 52.78, 50, 47.22]),
            "FALSE": clean([42.22, 40.74, 42.47, 43.95, 43.21, 45.68]),
        },
        "global_first": {
            "TRUE":  clean([38.89, 41.67, 44.44, 52.78, 50, 47.22]),
            "FALSE": clean([42.22, 40.74, 42.47, 43.95, 43.21, 45.68]),
        },
    }
}

# -----------------------
# PLOTTING
# -----------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)

for ax, (dataset, dataset_data) in zip(axes, data.items()):
    for doc in doc_classes:
        d = dataset_data[doc]

        # TRUE = solid
        ax.plot(p_vals, d["TRUE"],
                linestyle='-',
                marker='o',
                color=colors[doc],
                label=f"{doc} (TRUE)" if dataset == "Civic" else "")

        # FALSE = dashed
        ax.plot(p_vals, d["FALSE"],
                linestyle='--',
                marker='o',
                color=colors[doc],
                label=f"{doc} (FALSE)" if dataset == "Civic" else "")

    ax.set_title(dataset)
    ax.set_xlabel("p (%)")
    ax.grid(True)

axes[0].set_ylabel("Accuracy (%)")

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5)

plt.tight_layout(rect=[0, 0.1, 1, 1])
# plt.show()
plt.savefig("acc_breakdown_by_doc_class_plot.png")