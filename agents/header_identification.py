import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
# set logging level as info
logging.getLogger().setLevel(logging.WARNING)
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_result_path, get_simple_qasper_docs, extract_numbers, get_unanswerable_qasper_qids, get_notmentioned_contract_qids
from eval.utils import global_normalize_answer, get_gold_answers

from config import DATASET_LIST, DATA_ROOT_FOLDER
from structured_rag.utils import get_nondummy_ancestors, get_textspan
from agents.utils import get_toc_level
import traceback


def load_headers(dataset, file_name, sht_type):
    greptext_clean_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_greptext_clean",
        file_name + ".json"
    )
    m_header_cnt = dict()
    if not os.path.exists(greptext_clean_path):
        logging.warning(f"Grep text clean not found for {dataset}/{file_name} ({sht_type}), path: {greptext_clean_path}")
        return m_header_cnt
    with open(greptext_clean_path, "r") as f:
        m_np_header_ts = json.load(f)

    
        
    for np, header_ts in m_np_header_ts.items():
        if sht_type != 'llm_txt_sht' and sht_type != 'llm_vision_sht':
            if np == "1":
                continue
        header = header_ts['heading']
        # clean heading: only keep alp and digit, and convert to lower case, remove all other chars including spaces
        header_clean = re.sub(r'[^a-zA-Z0-9]', '', header).lower()
        m_header_cnt[header_clean] = m_header_cnt.get(header_clean, 0) + 1
    return m_header_cnt

def header_identification_per_doc(dataset, file_name, sht_type):
    m_header_cnt = load_headers(dataset, file_name, sht_type)
    true_header_cnt = load_headers(dataset, file_name, "intrinsic")

    # common keys, get min
    match_cnt = 0
    for header_clean, cnt in true_header_cnt.items():
        if header_clean in m_header_cnt:
            match_cnt += min(cnt, m_header_cnt[header_clean])
    
    header_cnt = sum(m_header_cnt.values())
    true_header_cnt = sum(true_header_cnt.values())

    if true_header_cnt == 0:
        if header_cnt == 0:
            recall = 1.0
        else:
            recall = 0.0

    else:
        recall = match_cnt / true_header_cnt
    
    if header_cnt == 0:
        if true_header_cnt == 0:
            precision = 1.0
        else:
            precision = 0.0

    else:
        precision = match_cnt / header_cnt
    


    f1 = 0.0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)
    # print(f"{dataset} {file_name} {sht_type}: recall={recall:.2f}, precision={precision:.2f}, f1={f1:.2f}")
    return recall, precision, f1

def avg_header_identification_per_dataset(dataset, sht_type):
    if dataset == "civic_rand_v1":
        start_id = 0
        end_id = 107
    elif dataset == 'finance_rand_v1':
        start_id = 0
        end_id = 100
    elif dataset == 'contract_rand_v0_1':
        start_id = 0
        end_id = 248
    elif dataset == 'qasper_rand_v1':
        start_id = 0
        end_id = 500
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    recall_list = []
    precision_list = []
    f1_list = []
    for file_id in range(start_id, end_id):
        file_name = str(file_id)
        recall, precision, f1 = header_identification_per_doc(dataset, file_name, sht_type)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)


    avg_recall = sum(recall_list) / len(recall_list)
    avg_precision = sum(precision_list) / len(precision_list)
    avg_f1 = sum(f1_list) / len(f1_list)
    return avg_recall, avg_precision, avg_f1

if __name__ == "__main__":
    # print as csv, with columns: sht_type,dataset_name1,dataset_name2,...

    sht_type_list = [
        'grobid',
        'llm_txt_sht',
        'llm_vision_sht',
        ''
    ]

    m_sht_type_metric = dict()
    for sht_type in sht_type_list:

        recall_list = []
        precision_list = []
        f1_list = []

        for dataset in DATASET_LIST:
            avg_recall, avg_precision, avg_f1 = avg_header_identification_per_dataset(dataset, sht_type)
            recall_list.append(avg_recall)
            precision_list.append(avg_precision)
            f1_list.append(avg_f1)

        m_sht_type_metric[sht_type] = {
            'recall': recall_list, # by dataset order in DATASET_LIST
            'precision': precision_list,
            'f1': f1_list,
        }

    # print as csv
    for metric in ['recall', 'precision', 'f1']:
        print(f"sht_type,{','.join(DATASET_LIST)}")
        for sht_type in sht_type_list:
            metric_values = m_sht_type_metric[sht_type][metric]
            metric_values_str = ','.join([f"{v:.2f}" for v in metric_values])
            print(f"{sht_type},{metric_values_str}")
        


    