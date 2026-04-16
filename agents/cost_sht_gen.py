import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name
import pandas as pd
import matplotlib.pyplot as plt
logging.getLogger().setLevel(logging.INFO)

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_baseline_messages, get_llm_response, pretty_repr, get_doc_txt, get_result_path, get_simple_qasper_docs, get_cost_usd
from agents.accuracy import INVALID_CIVIC_QUERY_IDS
from agents.cost import get_cost_usd


def total_tree_gen_cost(dataset, method, model, sht_type):
    doc_gen_path = str(get_result_path(dataset, model, "baseline", sht_type)).replace("/core/", "/llm_gen_toc_response/").replace("baseline", sht_type.replace("_sht", ""))
    print(doc_gen_path)
    m_file_cost = dict()
    with open(doc_gen_path, 'r') as f:
        for line in f:
            doc_gen_result = json.loads(line)
            filename = doc_gen_result['file_name']
            input_tokens = doc_gen_result['input_tokens']
            cached_tokens = doc_gen_result['cached_tokens']
            output_tokens = doc_gen_result['output_tokens']
            if filename not in m_file_cost:
                m_file_cost[filename] = 0
            m_file_cost[filename] += get_cost_usd(model, input_tokens + cached_tokens, 0, output_tokens)

    file_gen_cost_sum = sum(m_file_cost.values())
    
    # result_path = get_result_path(dataset, model, method, sht_type)
    # qids = []
    # with open(result_path, 'r') as f:
    #     for line in f:
    #         result = json.loads(line)
    #         qids.append(result['id'])
    
    # queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
    # with open(queries_path, 'r') as f:
    #     queries = json.load(f)
    # m_qid_filename = {qinfo['id']: qinfo['file_name'] for qinfo in queries}

    # file_gen_cost_sum = 0
    # calculated_files = set()
    # for qid in qids:
    #     filename = m_qid_filename[qid]
    #     if filename in calculated_files:
    #         continue
    #     file_gen_cost_sum += m_file_cost[filename]
    #     calculated_files.add(filename)
    # logging.info(f"{dataset} ({method}, {sht_type}): {len()} queries evaluated.")
    print(f"{dataset} ({method}, {sht_type}): {len(m_file_cost)} documents evaluated.")
    return file_gen_cost_sum

if __name__ == "__main__":
    sht_gen_model = "gpt-5.4"
    for dataset in DATASET_LIST:
        for method in ["baseline"]:
            for sht_type in ["llm_vision_sht"]:
                cost = total_tree_gen_cost(dataset, method, sht_gen_model, sht_type)
                logging.info(f"Total tree generation cost for {dataset} ({method}, {sht_type}): ${cost:.4f}")