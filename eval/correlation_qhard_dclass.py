import logging.config
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
logging.disable(level=logging.INFO)

import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import eval.eval_finance
import pandas as pd


def get_hard_level_civic(q):
    if q['id'] in {384, 388, 391, 396, 400, 401, 404, 407, 409, 411}:
        return 1
    if q['id'] in [did * 20 + qid for did in range(19) for qid in range(10, 20)]:
        return 3
    # if q['id'] in [did * 20 + qid for did in range(19) for qid in range(10)] + [qid for qid in range(380, 418) if qid not in {384, 388, 391, 396, 400, 401, 404, 407, 409, 411}]:
    #     return 4
    return 4
    


def get_hard_leve(q):
    if q['hard_level'] in {0} and len(q['clusters']) > 0:
        return 4
    elif q['hard_level'] in {0} and len(q['clusters']) == 0:
        return 1
    return int(q['hard_level'])

def acc_breakdown_by_doc_class(dataset, context_config):
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

    # classify the queries by doc classes
    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), dataset, "queries.json")
    with open(queries_path, "r") as f:
        all_queries = json.load(f)
    assert [q['id'] for q in all_queries] == list(range(len(all_queries)))

    queries = [q for q in all_queries if q['id'] in answer_id_list]
    query_hardness_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_demanding_questions", f"{dataset}.jsonl")
    if dataset == "civic":
        for q in queries:
            q['hard_level'] = get_hard_level_civic(q)
    else:
        with open(query_hardness_path, "r") as f:
            query_hardness_info = [json.loads(line) for line in f]
        query_hardness_info = [qh for qh in query_hardness_info if qh['id'] in answer_id_list]
        for q, qh in zip(queries, query_hardness_info):
            assert q['id'] == qh['id'], f"Query ID mismatch between queries and hardness info for query ID {q['id']} in dataset {dataset}."
            q['hard_level'] = get_hard_leve(qh)


    assert len(queries) == len(accuracy_list), f"Number of queries ({len(queries)}) and accuracy list ({len(accuracy_list)}) don't match for dataset {dataset} and context config {context_config}."
    

    doc_classes = {
        "well_formatted": [],
        "loosely_formatted": [],
        # "depth_aligned": "depth_aligned",
        "local_first": [],
        # "global_first": "global_first"
    }
    for doc_class in doc_classes.keys():
        doc_class_breakdown_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"doc_class_{doc_class}_{dataset}.json")
        if not os.path.exists(doc_class_breakdown_file):
            raise ValueError(f"Doc class breakdown file {doc_class_breakdown_file} not found, skipping...")
        with open(doc_class_breakdown_file, "r") as f:
            doc_is_class = json.load(f)
        doc_classes[doc_class] = [doc for doc, is_class in doc_is_class.items() if is_class == True]
    
    m_qhard_dclass_acc = {
        tuple([dclass, qhard]): []
        for dclass in doc_classes.keys()
        for qhard in range(1, 5)
    }

    for qinfo, acc in zip(queries, accuracy_list):
        doc = qinfo['file_name']
        hard_level = qinfo['hard_level']
        for doc_class, doc_list in doc_classes.items():
            if doc in doc_list:
                m_qhard_dclass_acc[(doc_class, hard_level)].append(acc)
    
    factor = 1
    if dataset in ['qasper', 'finance']:
        factor = 3
    m_qhard_dclass_avg_acc = {
        k: sum(v)/(factor*len(v)) if len(v) > 0 else "null"
        for k, v in m_qhard_dclass_acc.items()
    }

    m_qhard_dclass_qnum = {
        k: len(v)
        for k, v in m_qhard_dclass_acc.items()
    }
    # print(m_qhard_dclass_qnum)

    # print(m_qhard_dclass_acc)

    return m_qhard_dclass_avg_acc, m_qhard_dclass_qnum

if __name__ == "__main__":

    dataset_list = ["civic", "contract", "qasper", "finance"]


    context_config = ("sht", None, "sbert", True, True, True, 0.2)

    for dataset in dataset_list:
        print(f"Dataset: {dataset}")
        m_qhard_dclass_acc, m_qhard_dclass_qnum = acc_breakdown_by_doc_class(dataset, context_config)
        # print a 2d table, row = doc_class, col = hard_level
        doc_classes = sorted(set([k[0] for k in m_qhard_dclass_acc.keys()]))
        hard_levels = sorted(set([k[1] for k in m_qhard_dclass_acc.keys()]))

        acc_table = pd.DataFrame(
            [[m_qhard_dclass_acc[(doc_class, hard_level)] for hard_level in hard_levels] for doc_class in doc_classes],
            index=doc_classes,
            columns=hard_levels
        )

        print(acc_table.to_csv(sep=','))

        qnum_table = pd.DataFrame(
            [[m_qhard_dclass_qnum[(doc_class, hard_level)] for hard_level in hard_levels] for doc_class in doc_classes],
            index=doc_classes,
            columns=hard_levels
        )

        # print(qnum_table.to_csv(sep=','))


        # # print using comma as separator
        # print(table.to_csv(sep=','))

        # #plot a heatmap using seaborn
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 6))
        # # remember to convert "null" to np.nan for better visualization
        # table = table.replace("null", float("nan"))
        # # color range should be between min and max of the table values
        # min_val = table.min().min()
        # max_val = table.max().max()
        # sns.heatmap(table, annot=True, cmap="YlGnBu", vmin=min_val, vmax=max_val)


        # plt.title(f"Accuracy by Doc Class and Query Hardness for {dataset}")
        # plt.xlabel("Query Hardness Level")
        # plt.ylabel("Document Class")
        # plt.savefig(f"correlation_qhard_dclass_{dataset}.png")
