import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_toc_numbered, get_doc_txt, get_toc_in_context_messages, get_llm_response, pretty_repr, get_result_path

MAX_OUTPUT_TOKENS = 150

DEBUG = True

def run_toc_in_context_per_query(dataset, qinfo, model, sht_type):
    doc_txt = get_doc_txt(dataset, qinfo["file_name"])
    toc_numbered_txt = get_toc_numbered(dataset, qinfo["file_name"])
    query = qinfo["query"]
    messages = get_toc_in_context_messages(dataset, query, doc_txt, toc_numbered_txt)
    if DEBUG == True:
        print(pretty_repr(messages))
        print("---" * 20)
    else:
        response = get_llm_response(messages, model)
        response['id'] = qinfo["id"]
        return response

if __name__ == "__main__":
    for sht_type in [
        'deep',
        'wide',
        'grobid',
        'shed'
    ]:
        for model in ['gpt-5.4', 
                    #   'gpt-5-mini'
                    ]:
            for dataset in DATASET_LIST[:1]:
                print(f"toc_in_context: {dataset}")
                queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
                with open(queries_path, 'r') as file:
                    queries = json.load(file)
                
                num_queries = int(len(queries) * 0.2)

                result_jsonl_path = get_result_path(dataset, model, "toc_in_context")

                if dataset == "civic":
                    start_id = 0
                    end_id = len(queries)
                elif dataset == 'finance':
                    start_id = 0
                    end_id = 74
                else:
                    start_id = 0
                    end_id = num_queries

                for qinfo in queries[:1]:
                    print(f"\tquery id: {qinfo['id']}")
                    result = run_toc_in_context_per_query(dataset, qinfo, model)
                    if DEBUG == False:
                        with open(result_jsonl_path, 'a') as file:
                            contents = json.dumps(result) + "\n"
                            file.write(contents)
