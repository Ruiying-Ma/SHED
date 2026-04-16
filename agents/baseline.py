import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_baseline_messages, get_llm_response, pretty_repr, get_result_path, get_doc_txt
import fitz
import unicodedata
import string
from agents.listing import strip_leading_numbering

# MAX_OUTPUT_TOKENS = 150

DEBUG = False
def normalize_string(text: str):
    return unicodedata.normalize('NFKC', text)

def white_space_fix(text):
    return " ".join(text.split())

def replace_punc(text, rep = " "):
    exclude = set(string.punctuation)
    # replace punctuation with space
    return "".join(ch if ch not in exclude else rep for ch in text)


def get_txt(dataset, qinfo):
    assert dataset in ['contract_rand_v0_1', 'finance_rand_v1', 'qasper_rand_v1']
    with open(Path(DATA_ROOT_FOLDER) / dataset / "intrinsic" / "toc_greptext_clean" / (qinfo["file_name"] + ".json"), 'r') as file:
        toc_greptext = json.load(file)
    doc_txt = ""
    for level_str, text_span in toc_greptext.items():
        heading = text_span['heading']
        contxt = text_span['text']
        if heading.strip() != "":
            doc_txt += f"{heading}\n"
        if contxt.strip() != "":
            doc_txt += f"{contxt}\n"

    return doc_txt
    


def run_baseline_per_query(dataset, qinfo, model, reasoning_effort):
    if dataset != 'civic_rand_v1':
        doc_txt = get_txt(dataset, qinfo)
    else:
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

        for dataset in DATASET_LIST[3:]:
            print(f"Baseline: {dataset}")
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)
            
            result_jsonl_path = get_result_path(dataset, model, "baseline", 'intrinsic')

            if dataset == "civic_rand_v1":
                start_id = 0
                end_id = len(queries)
            elif dataset == 'finance_rand_v1':
                start_id = 74
                end_id = 100
            elif dataset == 'contract_rand_v0_1':
                start_id = 0
                end_id = 248
            elif dataset == 'qasper_rand_v1':
                start_id = 290
                end_id = 500
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            for qinfo in queries[start_id:end_id]:
                print(f"\tquery id: {qinfo['id']}")
                result = run_baseline_per_query(dataset, qinfo, model, reasoning_effort)
                if DEBUG == True:
                    continue
                with open(result_jsonl_path, 'a') as file:
                    contents = json.dumps(result) + "\n"
                    file.write(contents)
                