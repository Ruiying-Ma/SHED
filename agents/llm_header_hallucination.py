import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
from typing import TypedDict
import time
import pickle
from dotenv import load_dotenv
import traceback
import re

from config import DATASET_LIST, DATA_ROOT_FOLDER, get_max_window
from agents.utils import get_toc_system_message, get_system_message, get_timestamp, get_result_path, LLMResponse



def llm_raw_header_num(dataset, file_name, sht_type):
    sht_gen_path = f"/home/ruiying/SHTRAG/agents/results/llm_gen_toc_response/gpt-5.4/{sht_type.replace('_sht', '')}/{dataset}.jsonl"
    raw_md_str_list = []
    with open(sht_gen_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            if r['file_name'] == file_name:
                raw_md_str_list.append(r['message'])
        
    if len(raw_md_str_list) == 0:
        return 0
        
    md_lines = []
    for raw_md_str in raw_md_str_list:
        if raw_md_str.strip().startswith("```markdown") and raw_md_str.strip().endswith("```"):
            md_str = raw_md_str.strip()[len("```markdown"): -len("```")]
        else:
            md_str = raw_md_str
        md_lines += md_str.splitlines()
    
    header_list = []
    cur_line = ""
    for line in md_lines:
        if line.startswith("#"):
            if cur_line.startswith("#"):
                header_list.append(cur_line)
            cur_line = line
        else:
            cur_line += line
    if cur_line.startswith("#"):
        header_list.append(cur_line)

    return len(header_list)

def llm_extracted_header_num(dataset, file_name, sht_type):
    toc_textspan_path = f"/home/ruiying/SHTRAG/data/{dataset}/{sht_type}/toc_textspan_clean/{file_name}.json"
    if not os.path.exists(toc_textspan_path):
        logging.warning(f"TOC textspan not found for {dataset}/{file_name} ({sht_type}), path: {toc_textspan_path}")
        return 0
    else:
        with open(toc_textspan_path, 'r') as f:
            toc_textspan = json.load(f)
        return len(toc_textspan)
    
def hallucination_per_dataset(dataset, sht_type):
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

    hall_rate_list = []
    for file_id in range(start_id, end_id):
        file_name = str(file_id)
        raw_header_num = llm_raw_header_num(dataset, file_name, sht_type)
        extracted_header_num = llm_extracted_header_num(dataset, file_name, sht_type)
        if raw_header_num == 0:
            logging.warning(f"No raw header found for {dataset}/{file_name} ({sht_type}), skipping hallucination rate calculation for this file.")
            continue
        hallucination_rate = (raw_header_num - extracted_header_num) / raw_header_num
        hall_rate_list.append(hallucination_rate)
    avg_hall_rate = sum(hall_rate_list) / len(hall_rate_list)
    print(f"{dataset} ({sht_type}): Average hallucination rate: {avg_hall_rate:.3f}")


if __name__ == "__main__":
    sht_types = [
        'llm_txt_sht',
        'llm_vision_sht'
    ]
    for dataset in DATASET_LIST:
        for sht_type in sht_types:
            hallucination_per_dataset(dataset, sht_type)