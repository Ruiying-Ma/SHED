import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import traceback
import json
import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import eval.eval_finance

H=3
LABEL_RATIO = 5
TICK_RATIO = 4
TITLE_RATIO = 5.25

def get_eval_result_structurally_demanding(context_config, dataset, metric):
    if dataset == "civic":
        hard_q_ids = sorted([did * 20 + qid for did in range(19) for qid in range(10)] + [qid for qid in range(380, 418) if qid not in {384, 388, 391, 396, 400, 401, 404, 407, 409, 411}])
        # hard_q_ids = range(0, 418)
    else:
        sturct_demanding_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_demanding_questions", f"{dataset}.jsonl")
        with open(sturct_demanding_path, 'r') as file:
            hardness_info_list = [json.loads(line) for line in file.readlines()]
        hard_q_ids = [qinfo['id'] for qinfo in hardness_info_list if qinfo['hard_level'] in {0} and len(qinfo['clusters']) > 0]


    print(f"Dataset {dataset}: total {len(hard_q_ids)} structurally demanding questions.")
    if dataset == "civic":
        acc_list_1 = eval.eval_civic.civic_q1_eval_answer_list(context_config)
        _, _, acc_list_2 = eval.eval_civic.civic_q2_eval_answer_list(context_config)
        assert len(acc_list_1) == 380
        assert len(acc_list_2) == 38
        acc_list = acc_list_1 + acc_list_2
    elif dataset == "contract":
        acc_list, answer_id_list = eval.eval_contract.contract_eval_answer_list(context_config, False)
        assert all([aid in answer_id_list for aid in hard_q_ids])
        assert len(acc_list) == 1241
    elif dataset == "qasper":
        rating_list = eval.eval_qasper.qasper_eval_answer_llm_list(context_config)
        acc_list = [r / 3 for r in rating_list]
        assert len(acc_list) == 1451
    elif dataset == "finance":
        rating_list = eval.eval_finance.finance_eval_answer_llm_list(context_config)
        acc_list = [r / 3 for r in rating_list]
        assert len(acc_list) == 150
    
    
    hard_acc_list = [acc_list[qid] for qid in hard_q_ids]
    avg_acc = round(sum(hard_acc_list) * 100 / len(hard_acc_list), 3)
    return avg_acc

    

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

def set_subfigure(ax, dataset, need_title, need_xlabel, need_ylabel):
    if need_xlabel == True:
        x = config.CONTEXT_LEN_RATIO_LIST
        ax.set_xticks(x)
        ax.set_xticklabels([int(xi * 100) for xi in x])
        ax.set_xlim(min(x), max(x))

    
    if need_title == True:
        ax.set_title(rename_dataset(dataset), fontsize=H * TITLE_RATIO, fontstyle="italic")
    
    if need_xlabel == True:
        ax.set_xlabel("p (%)", fontsize=H * LABEL_RATIO)
        ax.tick_params(axis='x', labelsize=H * TICK_RATIO)
    
    if need_ylabel != None:
        ax.set_ylabel("accuracy (%)", fontsize=H * LABEL_RATIO)
        ax.tick_params(axis='y', labelsize=H * TICK_RATIO)
        x_current, y_current = ax.yaxis.label.get_position()
        ax.yaxis.set_label_coords(x_current - H * 0.075, need_ylabel)
    
    ax.grid(True, color="lightgray", linewidth=H * 0.25)
    
    ax.set_facecolor("whitesmoke")


def plot_e2e(
        baseline_list = ["sht", "vanilla", "raptor", "hipporag", "graphrag"],
        dataset_list=["civic", "contract", "qasper", "finance"],
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


    # load data
    m_dataset_baseline_yvalues = dict() # dataset -> baseline -> [accuracy for p in top_p]
    for dataset in dataset_list:
        assert dataset not in m_dataset_baseline_yvalues
        m_dataset_baseline_yvalues[dataset] = dict()
        metric = None
        if "civic" in dataset:
            metric = civic_metric
        if dataset == "qasper":
            metric = qasper_metric
        for baseline in baseline_list:
            assert baseline not in m_dataset_baseline_yvalues[dataset]
            m_dataset_baseline_yvalues[dataset][baseline] = [
                get_eval_result_structurally_demanding((baseline, None, embedding_model, True, True, True, p), dataset, metric)
                for p in config.CONTEXT_LEN_RATIO_LIST
            ]

    for dataset in dataset_list:
        for baseline in baseline_list:
            csv_line = dataset + "," + baseline
            for v in m_dataset_baseline_yvalues[dataset][baseline]:
                csv_line += f",{v}"
            print(csv_line)

    def plot_subfigure(d, ax):
        for bid, baseline in enumerate(baseline_list):
            style = m_baseline_style[baseline]
            color = style[0]
            shape = style[1]
            width = H * 0.75
            ax.plot(config.CONTEXT_LEN_RATIO_LIST, m_dataset_baseline_yvalues[d][baseline], color=color, linestyle=shape, linewidth=width, zorder=100-bid, label=rename_baseline(baseline))

    # Create a 1x grid of subplots
    fig = plt.figure(figsize=(H * len(dataset_list) * 0.8, H * 0.8))
    height_ratio = 9
    gs = fig.add_gridspec(2, len(dataset_list), height_ratios=[height_ratio, 1])
    axes = []
    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        if "civic" in dataset:
            ax = fig.add_subplot(gs[:, i])
            plot_subfigure(dataset, ax)

            # # TODO: add other fig config here
            # y_interval = 4
            # y_min = 51
            # num_y_interval = 5
            # ax.set_yticks([y_min + i * y_interval for i in range(num_y_interval + 1)])
            # ax.set_ylim(y_min - y_interval / 1.5, y_min + y_interval * (num_y_interval) + y_interval / 1.5)
            # ax.tick_params(axis='both', labelsize=H * TICK_RATIO)

            set_subfigure(ax, dataset, True, True, 0.5)
            axes.append(ax)
        else:
            ax_top = fig.add_subplot(gs[0, i])
            ax_bottom = fig.add_subplot(gs[1, i], sharex=ax_top)

            plot_subfigure(dataset, ax_top)
            plot_subfigure(dataset, ax_bottom)

            # TODO: add other fig config here
            graphrag_y = m_dataset_baseline_yvalues[dataset]["graphrag"][0]
            ax_bottom.set_ylim(graphrag_y * 0.9, graphrag_y * 1.1)
            all_other_y = [v for b in baseline_list if b != "graphrag" for v in m_dataset_baseline_yvalues[dataset][b]]
            ax_top.set_ylim(min(all_other_y) * 0.9, max(all_other_y) * 1.1)
            # hide connecting spine
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_bottom.xaxis.tick_bottom()
            # add diagonal "break" marks
            d = H * 0.005
            dd = 1
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)        # top-left
            ax_top.plot((dd-d, dd+d), (-d, +d), **kwargs)      # top-right
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (dd-d, dd+d), **kwargs)   # bottom-left
            ax_bottom.plot((dd-d, dd+d), (dd-d, dd+d), **kwargs) # bottom-right

            # # TODO: add other fig config here
            # if dataset == "contract":
            #     y_interval = 6
            #     y_min = 57
            #     num_y_interval = 4
            #     ax_top.set_yticks([y_min + i * y_interval for i in range(num_y_interval + 1)])
            #     ax_top.set_ylim(y_min - y_interval / 1.5, y_min + y_interval * (num_y_interval) + y_interval / 1.5)
            #     ax_top.tick_params(axis='both', labelsize=H * TICK_RATIO)
            #     y_mid = 7
            #     ax_bottom.set_yticks([y_mid])
            #     ax_bottom.set_ylim([y_mid - y_interval/2, y_mid + y_interval / 2])
            #     ax_bottom.tick_params(axis='both', labelsize=H * TICK_RATIO)
            
            # if dataset == "qasper":
            #     y_interval = 4
            #     y_min = 53
            #     num_y_interval = 4
            #     ax_top.set_yticks([y_min + i * y_interval for i in range(num_y_interval + 1)])
            #     ax_top.set_ylim(y_min - y_interval / 1.5, y_min + y_interval * (num_y_interval) + y_interval / 1.5)
            #     ax_top.tick_params(axis='both', labelsize=H * TICK_RATIO)
            #     y_mid = 30
            #     ax_bottom.set_yticks([y_mid])
            #     ax_bottom.set_ylim([y_mid - y_interval/2, y_mid + y_interval / 2])
            #     ax_bottom.tick_params(axis='both', labelsize=H * TICK_RATIO)

            # if dataset == "finance":
            #     y_interval = 2
            #     y_min = 36
            #     num_y_interval = 4
            #     ax_top.set_yticks([y_min + i * y_interval for i in range(num_y_interval + 1)])
            #     ax_top.set_ylim(y_min - y_interval / 1.5, y_min + y_interval * (num_y_interval) + y_interval / 1.5)
            #     ax_top.tick_params(axis='both', labelsize=H * TICK_RATIO)
            #     y_mid = 2
            #     ax_bottom.set_yticks([y_mid])
            #     ax_bottom.set_ylim([y_mid - y_interval/2, y_mid + y_interval / 2])
            #     ax_bottom.tick_params(axis='both', labelsize=H * TICK_RATIO)

            


            set_subfigure(ax_top, dataset, True, False, 0.39)
            set_subfigure(ax_bottom, dataset, False, True, None)
            ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            axes.append((ax_top, ax_bottom))

        

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(baseline_list),
        bbox_to_anchor=(0.505, 0.025),
        fontsize=H * TITLE_RATIO,        # adjust legend font size
        frameon=True     # cleaner look
    )

    plt.tight_layout(rect=[0.01, -0.05, 1, 1])
    # plt.show()
    plt.savefig("figure/end_to_end_hard.png", bbox_inches="tight")

if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)
    plot_e2e(
        # baseline_list = ["vanilla", "raptor", "graphrag", "sht"],
        dataset_list=["civic", "contract", "qasper"]
    )
