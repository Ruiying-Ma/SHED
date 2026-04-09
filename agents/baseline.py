import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_baseline_messages, get_llm_response, pretty_repr, get_doc_txt, get_result_path

# MAX_OUTPUT_TOKENS = 150

DEBUG = False

def run_baseline_per_query(dataset, qinfo, model, reasoning_effort):
    doc_txt = get_doc_txt(dataset, qinfo["file_name"])
    query = qinfo["query"]
    messages = get_baseline_messages(dataset, query, doc_txt)
    if DEBUG:
        print(pretty_repr(messages))
        print("---" * 20)
    else:
        response = get_llm_response(messages, model, reasoning_effort)
        response['id'] = qinfo["id"]
        return response

if __name__ == "__main__":
    # reasoning_effort = 'medium'
    reasoning_effort = None

    for model in [
        'gpt-5.4', 
        # 'gpt-5-mini'
    ]:

        for dataset in DATASET_LIST[:-1]:
            print(f"Baseline: {dataset}")
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)
            
            result_jsonl_path = get_result_path(dataset, model, "baseline", 'intrinsic')

            if dataset == "civic_rand_v1":
                start_id = 0
                end_id = len(queries)
            elif dataset == 'finance_rand_v1':
                start_id = 0
                end_id = 74
            elif dataset == 'contract_rand_v0_1':
                start_id = 0
                end_id = 248
            elif dataset == 'qasper':
                start_id = 0
                end_id = 290

            for qinfo in queries[start_id:end_id]:
                print(f"\tquery id: {qinfo['id']}")
                result = run_baseline_per_query(dataset, qinfo, model, reasoning_effort)
                if DEBUG == True:
                    continue
                with open(result_jsonl_path, 'a') as file:
                    contents = json.dumps(result) + "\n"
                    file.write(contents)
                