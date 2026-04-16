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


def acc_breakdown_by_doc_class(dataset, context_config, doc_class, is_stratified=False):
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


    assert len(queries) == len(accuracy_list), f"Number of queries ({len(queries)}) and accuracy list ({len(accuracy_list)}) don't match for dataset {dataset} and context config {context_config}."
    
    doc_class_breakdown_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"doc_class_{doc_class}_{dataset}.json")
    if not os.path.exists(doc_class_breakdown_file):
        raise ValueError(f"Doc class breakdown file {doc_class_breakdown_file} not found, skipping...")
    with open(doc_class_breakdown_file, "r") as f:
        doc_is_class = json.load(f)



    m_class_acc_list = {
        True: [],
        False: [],
        "other": []
    }

    if is_stratified == False:
        for qinfo, acc in zip(queries, accuracy_list):
            doc = qinfo['file_name']
            dclass = doc_is_class[doc]
            if dclass == True:
                m_class_acc_list[True].append(acc)
            elif dclass == False:
                m_class_acc_list[False].append(acc)
            else:
                m_class_acc_list["other"].append(acc)
        assert sum([len(v) for v in m_class_acc_list.values()]) == len(accuracy_list), f"Total number of queries in class breakdown ({sum([len(v) for v in m_class_acc_list.values()])}) doesn't match total number of queries ({len(accuracy_list)}) for dataset {dataset} and context config {context_config}."
    else:
        m_doc_acc_list = dict()
        for qinfo, acc in zip(queries, accuracy_list):
            doc = qinfo['file_name']
            if doc not in m_doc_acc_list:
                m_doc_acc_list[doc] = []
            m_doc_acc_list[doc].append(acc)
        for doc, acc_list in m_doc_acc_list.items():
            dclass = doc_is_class[doc]
            doc_avg_acc = sum(acc_list) / len(acc_list)
            if dclass == True:
                m_class_acc_list[True].append(doc_avg_acc)
            elif dclass == False:
                m_class_acc_list[False].append(doc_avg_acc)
            else:
                m_class_acc_list["other"].append(doc_avg_acc)
        assert sum([len(v) for v in m_class_acc_list.values()]) == len(doc_is_class), f"Total number of documents in class breakdown ({sum([len(v) for v in m_class_acc_list.values()])}) doesn't match total number of documents ({len(doc_is_class)}) for dataset {dataset} and context config {context_config}."


    # print(m_class_acc_list)
    # print(dataset, doc_class, {k: len(v) for k, v in m_class_acc_list.items()})


    factor = 1
    if dataset in ['qasper', 'finance']:
        factor = 3

    m_class_acc = {
        c: round(sum(acc_list) * 100 / (len(acc_list) * factor), 2) if len(acc_list) > 0 else None
        for c, acc_list in m_class_acc_list.items()
    }
    return m_class_acc[True], m_class_acc[False], m_class_acc["other"]
        

if __name__ == "__main__":
    doc_class_list = [
        'well_formatted',
        'loosely_formatted',
        'depth_aligned',
        'local_first',
        'global_first'
    ]

    dataset_list = ["civic", "contract", "qasper", "finance"]


    context_config = ("sht", None, "sbert", True, True, True, 0.2)

    for doc_class in doc_class_list:
        tab_str = f"{doc_class},"
        for dataset in dataset_list:
            class_true_acc, class_false_acc, class_other_acc = acc_breakdown_by_doc_class(dataset, context_config, doc_class, is_stratified=True)
            tab_str += f"{class_true_acc},{class_false_acc},{class_other_acc},"
        print(tab_str)
            