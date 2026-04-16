import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.rag_ablation import get_eval_result
import config
from eval.plot_e2e_old import rename_dataset, rename_baseline

H=3
LABEL_RATIO = 5
TICK_RATIO = 4
TITLE_RATIO = 5.25

def rename_sht(sht):
    if sht == "grobid":
        return "GROBID"
    elif sht == None:
        return "SHED"
    elif sht == "intrinsic":
        return "true"
    elif sht == "wide":
        return "Wide"
    elif sht == "deep":
        return "Deep"
    elif sht == "llm_txt":
        return "LLM-text"
    elif sht == "llm_vision":
        return "LLM-vision"
    else:
        raise ValueError(f"Unknown sht: {sht}")


## in one subplot
# def histogram_sht_ablation(
#         sht_list=["grobid", None, "intrinsic"],
#         dataset_list=["civic1", "civic2", "contract", "qasper"],
#         civic_metric="f1",
#         qasper_metric="llmjudge",
#         embedding_model="sbert",
#         p=0.2,
#     ):

#     m_sht_style = {
#         "grobid": ("green"),
#         None: ("red"),
#         "intrinsic": ("blue")
#     }

#     m_sht_perf = {
#         sht: [
#                 get_eval_result(
#                     context_config=("sht", sht, embedding_model, True, True, True, p),
#                     dataset=dataset,
#                     metric=civic_metric if "civic" in dataset else (qasper_metric if dataset == "qasper" else None)
#                 )
#                 for dataset in dataset_list
#             ]
#         for sht in list(set(sht_list + ["intrinsic"]))
#     }

#     m_sht_relative_perf = {
#         sht: [100 * (true_perf - perf)/true_perf for perf, true_perf in zip(m_sht_perf[sht], m_sht_perf["intrinsic"])]
#         for sht in m_sht_perf
#     }
    

#     fig, ax = plt.subplots(figsize=(H * 2, H))
#     bar_width = H/10
#     group_gap = H/8

#     # Compute the x positions for the first bar of each dataset
#     indices = np.arange(len(dataset_list)) * (len(sht_list) * bar_width + group_gap)

#     for sht_id, sht in enumerate(sht_list):
#         ax.bar(
#             indices + sht_id * bar_width,
#             m_sht_relative_perf[sht],
#             width=bar_width,
#             label=rename_sht(sht),
#             color=m_sht_style[sht][0],
#             zorder=3
#         )
    
#     ax.set_xticks(indices - (bar_width / 2) + ((len(sht_list) * bar_width) / 2))
#     ax.set_xticklabels([rename_dataset(d) for d in dataset_list], fontsize=H * 5.25, fontstyle="italic")
#     ax.set_ylabel("accuracy (%)", fontsize=H * 5)
#     ax.legend(fontsize=H * 5.25, frameon=True, ncol=3, loc='upper center')
#     # Style tweaks
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linewidth=H * 0.25, color="lightgray", zorder=0)
#     ax.set_facecolor("whitesmoke")

#     plt.tight_layout()
#     plt.show()


# in multiple subplot
def histogram_sht_ablation(
        sht_list=["llm_txt", "llm_vision", "grobid", "wide", "deep", None],
        dataset_list=["civic", "contract", "qasper", "finance"],
        civic_metric="f1",
        qasper_metric="llmjudge",
        embedding_model="sbert",
        p=0.2,
    ):

    m_sht_style = {
        "llm_txt": ("forestgreen",),
        "llm_vision": ("orange",),
        "grobid": ("blue",),
        "wide": ("purple",),
        "deep": ("pink",),
        None: ("red",),
        "intrinsic": ("forestgreen",)
    }

    m_sht_perf = {
        sht: [
                get_eval_result(
                    context_config=("sht", sht, embedding_model, True, True, True, p),
                    dataset=dataset,
                    metric=civic_metric if "civic" in dataset else (qasper_metric if dataset == "qasper" else None)
                )
                for dataset in dataset_list
            ]
        for sht in list(set(sht_list + ["intrinsic"]))
    }

    m_sht_relative_perf = {
        sht: [100 * (true_perf - perf)/true_perf for perf, true_perf in zip(m_sht_perf[sht], m_sht_perf["intrinsic"])]
        # sht: [(true_perf - perf) for perf, true_perf in zip(m_sht_perf[sht], m_sht_perf["intrinsic"])]
        for sht in m_sht_perf
    }

    ######################################################### print tab
    tab_str = ""
    for sht in sht_list:
        tab_str += rename_sht(sht)
        for dataset in dataset_list:
            perf = -m_sht_relative_perf[sht][dataset_list.index(dataset)]
            tab_str += f",{perf:.2f}"
        tab_str += "\n"
    print(tab_str)

    #########################################################
    

    fig, axes = plt.subplots(1, len(dataset_list), figsize=(H * len(dataset_list) * 0.85, H*0.8), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax_idx, ax in enumerate(axes):
    
        bar_width = H/(10 * len(sht_list))
        bar_gap = H/(20 * len(sht_list))
        
        # Compute the x positions for the first bar of each dataset
        indices = np.arange(len(sht_list)) * (bar_width + bar_gap)

        for sht_id, sht in enumerate(sht_list):
            ax.bar(
                [indices[sht_id]],
                [m_sht_relative_perf[sht][ax_idx]],
                width=bar_width,
                color=m_sht_style[sht][0],
                label=rename_sht(sht),
                zorder=3,
                edgecolor="black",
            )

        dataset = dataset_list[ax_idx]
        y_interval = None
        if dataset == "civic":
            y_interval = 9
            ax.set_yticks(np.array([-1, 0, 1, 2, 3, 4]) * y_interval)
            ax.set_ylim(-y_interval * 1.25, 4.35 * y_interval)
        elif dataset == "contract":
            y_interval = 1.7
            ax.set_yticks(np.array([-1, 0, 1, 2, 3, 4]) * y_interval)
            ax.set_ylim(-y_interval * 1.25, 4.35 * y_interval)
        elif dataset == "qasper":
            y_interval = 2.6
            ax.set_yticks(np.array([-1, 0, 1, 2, 3, 4]) * y_interval)
            ax.set_ylim(-y_interval * 1.25, 4.35 * y_interval)
        elif dataset == "finance":
            y_interval = 10
            ax.set_yticks(np.array([-1, 0, 1, 2, 3, 4]) * y_interval)
            ax.set_ylim(-y_interval * 1.25, 4.35 * y_interval)
        
        
        ax.set_ylabel("accuracy reduction\nfrom true SHTs (%)", fontsize=H * LABEL_RATIO)
        ax.set_xticks([])
        ax.tick_params(axis='both', labelsize=H * TICK_RATIO)  # show ticks and adjust font size
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linewidth=H * 0.25, color="lightgray", zorder=0)
        ax.axhline(0, color="black", zorder=5, linewidth=H*0.25)
        ax.set_facecolor("whitesmoke")
        ax.set_title(rename_dataset(dataset), fontstyle="italic", fontsize=H*TITLE_RATIO)

    # Create one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=int(len(sht_list)),
        bbox_to_anchor=(0.51, -0),
        fontsize=H * TITLE_RATIO * 0.92,        # adjust legend font size
        frameon=True       # cleaner look
    )

    plt.tight_layout(rect=[0.01, -0.05, 1, 1])
    plt.savefig("figure/ablation_sht.pdf", bbox_inches="tight")



if __name__ == "__main__":
    histogram_sht_ablation(
        dataset_list=["civic1", "civic2", "contract", "qasper", "finance"]
    )