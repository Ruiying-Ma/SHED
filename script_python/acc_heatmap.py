import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import eval.eval_finance

def get_finance_data_dict():
    m_config_acc_list = {}
    vanilla_config = ("vanilla", None, "sbert", None, None, None, 0.2)
    raptor_config = ("raptor", None, "sbert", None, None, None, 0.2)
    graphrag_config = ("graphrag", None, None, None, None, None, None)
    hipporag_config = ("hipporag", None, "sbert", None, None, None, 0.2)
    shed_config = ("sht", None, "sbert", True, True, True, 0.2)
    orig_model_config = ("sht", "intrinsic", "sbert", True, True, True, 0.2)
    wide_config = ("sht", "wide", "sbert", True, True, True, 0.2)
    for context_config in [
        vanilla_config,
        orig_model_config,
        wide_config,
        shed_config,
        # raptor_config,
        # graphrag_config,
        # hipporag_config,
    ]:
        print(f"Evaluating context_config={context_config}...")
        acc_list = eval.eval_finance.finance_eval_answer_llm_list(context_config)
        acc_list = [0 if acc <= 1 else 1 for acc in acc_list]
        m_config_acc_list[str(context_config[:2])] = acc_list

    new_m_config_acc_list = dict()
    for idx in range(len(m_config_acc_list[str(orig_model_config[:2])])):
        if len(set(
            m_config_acc_list[k][idx]
            for k in m_config_acc_list
        )) > 1:
            for k in m_config_acc_list:
                if k not in new_m_config_acc_list:
                    new_m_config_acc_list[k] = []
                new_m_config_acc_list[k].append(m_config_acc_list[k][idx])
    return new_m_config_acc_list


def dict_list_to_heatmap(data_dict, figsize=(20, 6), title="Heatmap"):
    """
    Draw a heatmap where:
      - y-axis = dict keys
      - x-axis = list indices
      - cell value = float at that index for that key
    """
    keys = list(data_dict.keys())
    max_len = max((len(v) for v in data_dict.values()), default=0)

    # Build matrix (pad with NaN)
    matrix = np.full((len(keys), max_len), np.nan, dtype=float)
    for i, k in enumerate(keys):
        vals = data_dict[k]
        matrix[i, :len(vals)] = vals

    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, aspect='auto', interpolation='nearest')

    # Axis labels
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xticks(np.arange(max_len))
    ax.set_xlabel("List index")
    ax.set_ylabel("Keys")
    ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    # # Optional annotations
    # for i in range(masked.shape[0]):
    #     for j in range(masked.shape[1]):
    #         if not masked.mask[i, j]:
    #             ax.text(j, i, f"{masked[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"acc_heatmap.png")

if __name__ == "__main__":
    data_dict = get_finance_data_dict()
    dict_list_to_heatmap(data_dict, title="Finance Dataset LLM Ratings Heatmap")
