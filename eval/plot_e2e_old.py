import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_tab import get_eval_result
import config
import traceback

H=3

def rename_baseline(baseline):
    if baseline == "vanilla":
        return "Vanilla"
    elif baseline == "raptor":
        return "Raptor"
    elif baseline == "sht":
        return "SHED"
    elif baseline == "hipporag":
        return "HippoRAG"
    elif baseline == "graphrag":
        return "GraphRAG"
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

def rename_dataset(dataset):
    if dataset == "civic":
        return "Civic"
    elif dataset == "civic1":
        return "Civic1"
    elif dataset == "civic2":
        return "Civic2"
    elif dataset == "contract":
        return "ContractNLI"
    elif dataset == "qasper":
        return "Qasper"
    elif dataset == "finance":
        return "FinanceBench"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def plot_e2e(
        baseline_list = ["vanilla", "raptor", "hipporag", "graphrag", "sht"],
        dataset_list=["civic1", "civic2", "contract", "qasper"],
        civic_metric="f1",
        qasper_metric="llmjudge",
        embedding_model="sbert"
    ):

    m_baseline_style = {
        "vanilla": ('forestgreen', '-'),
        "raptor": ('orange', '-'),
        "hipporag": ('blue', '-'),
        "graphrag": ('purple', '-'),
        "sht": ('red', '--'),
    }


    # Create a 1x grid of subplots
    fig, axes = plt.subplots(1, len(dataset_list), figsize=(H * len(dataset_list) * 1.2, H), sharex=True, sharey=False)
    axes = axes.flatten()
    # Plot on each subplot
    for ax_idx, ax in enumerate(axes):
        dataset = dataset_list[ax_idx]
        metric = None
        if "civic" in dataset:
            metric = civic_metric
        if dataset == "qasper":
            metric = qasper_metric
        
        # min_y = None
        # max_y = None
        for baseline_id, baseline in enumerate(baseline_list):
            try:
                x = config.CONTEXT_LEN_RATIO_LIST
                y = [
                    get_eval_result((baseline, None, embedding_model, True, True, True, p), dataset, metric)
                    for p in config.CONTEXT_LEN_RATIO_LIST
                ]
                # local_min_y = min(y)
                # local_max_y = max(y)
                # if min_y == None or local_min_y < min_y:
                #     min_y = local_min_y
                # if max_y == None or local_max_y > max_y:
                #     max_y = local_max_y
                style = m_baseline_style[baseline]
                color = style[0]
                shape = style[1]
                width = H * 0.75
                ax.plot(x, y, label=rename_baseline(baseline), color=color, linestyle=shape, linewidth=width, zorder=100-baseline_id)
            except Exception as e:
                print(e)
                print(traceback.print_exc())
                continue

        # # set y ticks
        # ytick_low = int(min_y / 5) * 5
        # ytick_high = np.ceil(max_y / 5) * 5
        # ytick_num = 6
        # ax.set_yticks(np.linspace(ytick_low, ytick_high, num=ytick_num))
        # y_interval = (ytick_high - ytick_low) / (ytick_num - 1)
        # ax.set_ylim(ytick_low - y_interval/10, ytick_high + y_interval/10)
        y_interval = None
        y_min = None
        # if dataset == "civic":
        #     y_interval = 6
        #     y_min = 40
        # elif dataset == "contract":
        #     y_interval = 5
        #     y_min = 60
        # elif dataset == "qasper":
        #     y_interval = 3
        #     y_min = 55
        # elif dataset == "finance":
        #     y_interval = 2
        #     y_min = 37
        
        if y_interval != None:
            assert y_min != None
            num_y_interval = 5
            ax.set_yticks([y_min + i * y_interval for i in range(num_y_interval + 1)])
            ax.set_ylim(y_min, y_min + y_interval * (num_y_interval))

        ax.set_xticks(x)
        ax.set_xticklabels([int(xi * 100) for xi in x])
        ax.set_xlim(min(x), max(x))

        ax.set_title(rename_dataset(dataset), fontsize=H * 5.25, fontstyle="italic")
        ax.set_xlabel("top_p (%)", fontsize=H * 5)
        ax.set_ylabel("accuracy (%)", fontsize=H * 5)
        ax.tick_params(axis='both', labelsize=H * 4.5)  # show ticks and adjust font size
        ax.grid(True, color="lightgray", linewidth=H * 0.25)
        ax.set_facecolor("whitesmoke")

    # Create one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(baseline_list),
        bbox_to_anchor=(0.5, -0),
        fontsize=H * 5.25,        # adjust legend font size
        frameon=True       # cleaner look
    )

    # Shared labels
    # fig.supxlabel("X axis (shared)"
    # fig.supylabel("Accuracy (%)", fontsize=H * 6)

    plt.tight_layout(rect=[0.01, -0.05, 1, 1])
    plt.savefig("figure/end_to_end.pdf", bbox_inches="tight")

if __name__ == "__main__":
    plot_e2e(
        dataset_list=["civic", "contract", "qasper", "finance"]
    )
