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


def load_level(dataset, file_name, sht_type):
    greptext_clean_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_greptext_clean",
        file_name + ".json"
    )
    if not os.path.exists(greptext_clean_path):
        return None
    with open(greptext_clean_path, "r") as f:
        m_np_header_ts = json.load(f)
    if len(m_np_header_ts) == 0:
        return 0
    
    largest_level = max([len(np.split('.')) for np in m_np_header_ts.keys()])
    if sht_type != 'llm_txt_sht' and sht_type != 'llm_vision_sht':
        largest_level -= 1

    assert largest_level >= 0
    return largest_level
    

def level_per_doc(dataset, file_name, sht_type):
    pred_level = load_level(dataset, file_name, sht_type)

    if pred_level is None:
        return None

    return pred_level

def avg_level_per_dataset(dataset, sht_type):
    if dataset == "civic_rand_v1":
        start_id = 0
        end_id = 107
    elif dataset == 'finance_rand_v1':
        start_id = 0
        end_id = 74
    elif dataset == 'contract_rand_v0_1':
        start_id = 0
        end_id = 248
    elif dataset == 'qasper_rand_v1':
        start_id = 0
        end_id = 290
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    level_list = []
    for file_id in range(start_id, end_id):
        file_name = str(file_id)

        level = level_per_doc(dataset, file_name, sht_type)
        if level is not None:
            level_list.append(level)
        else:
            continue    


    avg_level = sum(level_list) / len(level_list)
    return avg_level

if __name__ == "__main__":
    # print as csv, with columns: sht_type,dataset_name1,dataset_name2,...

    sht_type_list = [
        # 'grobid',
        'llm_txt_sht',
        'llm_vision_sht',
        # '',
        'intrinsic',
    ]

    m_sht_type_metric = dict()
    for sht_type in sht_type_list:

        level_list = []
        for dataset in DATASET_LIST:
            avg_level = avg_level_per_dataset(dataset, sht_type)
            level_list.append(avg_level)


        m_sht_type_metric[sht_type] = {
            'level': level_list, # by dataset order in DATASET_LIST
        }

    # print as csv
    for metric in ['level']:
        print(f"sht_type,{','.join(DATASET_LIST)}")
        for sht_type in sht_type_list:
            metric_values = m_sht_type_metric[sht_type][metric]
            metric_values_str = ','.join([f"{v:.2f}" for v in metric_values])
            print(f"{sht_type},{metric_values_str}")
        


    