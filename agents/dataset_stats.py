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


def avg_doc_tokens_per_dataset(dataset, start_id, end_id):
    queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    total_tokens = []
    for qinfo in queries[start_id:end_id]:
        filename = qinfo['file_name']
        doc_txt = get_doc_txt(dataset, filename)
        tokens = doc_txt.split()
        total_tokens.append(len(tokens))
    
    avg_tokens = sum(total_tokens) / len(total_tokens)
    logging.info(f"{dataset}: {len(total_tokens)} documents, average tokens per document: {avg_tokens}")
    return avg_tokens

if __name__ == "__main__":
    for dataset in DATASET_LIST:
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
        avg_doc_tokens_per_dataset(dataset, start_id, end_id)