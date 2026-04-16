import logging.config
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
logging.disable(level=logging.INFO)
import matplotlib.pyplot as plt

import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import eval.eval_finance


def acc_f1_correlation(dataset, context_config):
    if dataset == "civic":
        accuracy_list = eval.eval_civic.civic_q1_eval_answer_list(context_config) + eval.eval_civic.civic_q2_eval_answer_list(context_config)[-1]
        answer_id_list = list(range(len(accuracy_list)))
    elif dataset == "contract":
        accuracy_list, answer_id_list = eval.eval_contract.contract_eval_answer_list(context_config)
    elif dataset == "qasper":
        accuracy_list = eval.eval_qasper.qasper_eval_answer_llm_list(context_config)
        answer_id_list = list(range(len(accuracy_list)))
    elif dataset == "finance":
        accuracy_list = eval.eval_finance.finance_eval_answer_llm_list(context_config)
        answer_id_list = list(range(len(accuracy_list)))
    else:
        raise ValueError(f"Dataset {dataset} not supported yet.")
    # accuracy_list is sorted by query id

    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset, "queries.json")
    with open(queries_path, "r") as f:
        all_queries = json.load(f)
    assert [q['id'] for q in all_queries] == list(range(len(all_queries)))

    queries = [q for q in all_queries if q['id'] in answer_id_list]


    assert len(queries) == len(accuracy_list), f"Number of queries ({len(queries)}) and accuracy list ({len(accuracy_list)}) don't match for dataset {dataset} and context config {context_config}."
    
    # get f1 of robustness and compactness
    top_down_f1_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"top_down_shed.csv")
    m_file_f1 = dict()
    is_header = True
    with open(top_down_f1_file, "r") as f:
        for l in f:
            if is_header == True:
                is_header = False
                continue
            fields = l.strip().split("::::::")
            assert len(fields) == 5, f"Unexpected number of fields ({len(fields)}) in line: {l}"
            if fields[0] != dataset:
                continue
            if fields[-1] == "None":
                m_file_f1[fields[1]] = None
            else:
                m_file_f1[fields[1]] = float(fields[-1])
    
    print(f"{dataset} #files = {len(m_file_f1)}")

    m_file_acc = dict()
    
    for qinfo, acc in zip(queries, accuracy_list):
        file_name = qinfo['file_name']
        if file_name not in m_file_acc:
            m_file_acc[file_name] = []
        m_file_acc[file_name].append(acc)

    m_file_avg_acc = dict()
    factor = 1
    for f, acc_list in m_file_acc.items():
        if dataset in ["qasper", "finance"]:
            factor = 3
        m_file_avg_acc[f] = sum(acc_list) / (factor * len(acc_list))

    sorted_file_list = sorted([f for f in m_file_f1 if m_file_f1[f] != None and f in m_file_avg_acc])
    x_list = [m_file_f1[f] for f in sorted_file_list]
    y_list = [m_file_avg_acc[f] for f in sorted_file_list]
    # plt.clf()
    # plt.plot(x_list, y_list, 'o', ms=1)
    # plt.xlabel("F1 of Robustness and Compactness")
    # plt.ylabel("Accuracy")
    # plt.title(f"{dataset}")
    # plt.savefig(f"acc_robustness_compactness_f1_{dataset}.png")

    # a histogram
    # x = bins, [0, 100], width=10, f1 of robustness and compactness
    # y = acc, average accuracy of files in the bin
    bins = [i for i in range(0, 101, 10)]
    bin_acc_list = {b: [] for b in bins}
    for f in sorted_file_list:
        f1 = m_file_f1[f] * 100
        acc = m_file_avg_acc[f]
        bin_id = max([b for b in bins if f1 >= b])
        bin_acc_list[bin_id].append(acc)
    x = []
    y = []
    for b in bins:
        if len(bin_acc_list[b]) > 0:
            x.append(b)
            y.append(sum(bin_acc_list[b]) / len(bin_acc_list[b]))
    plt.clf()
    plt.bar(x, y, width=5)
    plt.xlabel("F1 of Robustness and Compactness")
    plt.ylabel("Average Accuracy")
    plt.title(f"{dataset}")
    plt.savefig(f"acc_robustness_compactness_f1_hist_{dataset}.png")
    

        

if __name__ == "__main__":
    dataset_list = ["civic", "contract", "qasper", "finance"]
    context_config = ("sht", None, "sbert", True, True, True, 0.2)

    for dataset in dataset_list:
        acc_f1_correlation(dataset, context_config)