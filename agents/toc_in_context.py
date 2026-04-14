import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_toc_numbered_clean, get_doc_txt, get_toc_in_context_messages, get_llm_response, pretty_repr, get_result_path, LLMResponse
from agents.accuracy import INVALID_CIVIC_QUERY_IDS
MAX_OUTPUT_TOKENS = 150

DEBUG = False

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

def run_toc_in_context_per_query(dataset, qinfo, model, sht_type):
    if dataset != "civic_rand_v1":
        doc_txt = get_txt(dataset, qinfo)
    else:
        doc_txt = get_doc_txt(dataset, qinfo["file_name"])
    try:
        toc_numbered_txt = get_toc_numbered_clean(dataset, qinfo["file_name"], sht_type)
    except Exception as e:
        response = LLMResponse(
            is_success=False,
            message=f"{type(e).__name__}: {str(e)}",
            latency=0.0,
            input_tokens=0,
            cached_tokens=0,
            output_tokens=0,
        )
        return response
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
        # 'deep',
        # 'wide',
        # 'grobid',
        # '',
        # 'llm_txt_sht',
        'intrinsic',
    ]:
        for model in [
            'gpt-5.4', 
                    #   'gpt-5-mini'
        ]:
            for dataset in DATASET_LIST[-1:]:
                print(f"toc_in_context ({model}, {sht_type}): {dataset}")
                queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
                with open(queries_path, 'r') as file:
                    queries = json.load(file)
                
                result_jsonl_path = get_result_path(dataset, model, "toc_in_context", sht_type)

                if dataset == "civic_rand_v1":
                    start_id = 0
                    end_id = len(queries)
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

                for qinfo in queries[start_id:end_id]:
                    result = run_toc_in_context_per_query(dataset, qinfo, model, sht_type)
                    if DEBUG == False:
                        with open(result_jsonl_path, 'a') as file:
                            contents = json.dumps(result) + "\n"
                            file.write(contents)
