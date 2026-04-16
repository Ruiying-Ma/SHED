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
import unicodedata
import string

def normalize_string(text: str):
    return unicodedata.normalize('NFKC', text)

def white_space_fix(text):
    return " ".join(text.split())

def replace_punc(text, rep = " "):
    exclude = set(string.punctuation)
    # replace punctuation with space
    return "".join(ch if ch not in exclude else rep for ch in text)

def lower(text):
    return text.lower()

def clean_txt(text):
    return replace_punc(white_space_fix(normalize_string(text))).lower()

def eval_textspan(ts, true_ts):
    
    # recall, precision, f1
    ts_tokens = set(ts.split())
    true_ts_tokens = set(true_ts.split())
    if len(true_ts_tokens) == 0:
        recall = 1.0 if len(ts_tokens) == 0 else 0.0
    else:
        recall = len(ts_tokens & true_ts_tokens) / len(true_ts_tokens)
    if len(ts_tokens) == 0:
        precision = 1.0 if len(true_ts_tokens) == 0 else 0.0
    else:
        precision = len(ts_tokens & true_ts_tokens) / len(ts_tokens)

    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return {"recall": recall, "precision": precision, "f1": f1}
    

def load_textspan(dataset, file_name, sht_type):
    textspan_clean_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_textspan_clean",
        file_name + ".json"
    )
    if not os.path.exists(textspan_clean_path):
        logging.warning(f"Textspan clean not found for {dataset}/{file_name} ({sht_type}), path: {textspan_clean_path}")
        return None
    greptext_clean_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_greptext_clean",
        file_name + ".json"
    )
    if not os.path.exists(greptext_clean_path):
        logging.warning(f"Greptext clean not found for {dataset}/{file_name} ({sht_type}), path: {greptext_clean_path}")
        return None
    with open(textspan_clean_path, "r") as f:
        m_np_ts = json.load(f)
    with open(greptext_clean_path, "r") as f:
        m_np_header = json.load(f)
    assert set(m_np_ts.keys()) == set(m_np_header.keys()), f"NP keys in textspan and greptext do not match for {dataset}/{file_name} ({sht_type})"
    # clean header and textspan
    m_np_header_ts = dict()
    for np in m_np_ts.keys():
        if sht_type != 'llm_txt_sht' and sht_type != 'llm_vision_sht':
            if np == "1":
                continue
        header = m_np_header[np]['heading']
        textspan = m_np_ts[np]
        m_np_header_ts[np] = dict()
        m_np_header_ts[np]['heading'] = header
        m_np_header_ts[np]['clean_heading'] = clean_txt(header)
        m_np_header_ts[np]['clean_text'] = clean_txt(textspan) # excluding heading
    return m_np_header_ts
    
def eval_sht_per_doc_topdown(dataset, file_name, sht_type):
    print(f"Evaluating {dataset}/{file_name} ({sht_type})")
    true_greptext = load_textspan(dataset, file_name, "intrinsic")
    pred_greptext = load_textspan(dataset, file_name, sht_type)

    if len(true_greptext) == 0:
        return None
    

    if pred_greptext is None:
        logging.warning(f"No predicted greptext found for {dataset}/{file_name} ({sht_type}), skipping evaluation")
        return 0, 0.0, 0.0


    dst_path = Path(__file__).resolve().parent / "results" / "sht_eval" / "top_down" / sht_type / dataset / (file_name + ".jsonl")
    
    # evaluate textspan for each np in true_greptext
    max_recall_list = []
    max_precision_list = []
    max_f1_list = []
    for np, true_header_ts in true_greptext.items():
        if np == "1":
            continue
        # check max recall, precision, f1 among all pred_greptext
        metric_info = {
            "true_np": np,
            "true_header": true_header_ts['heading'],
            "max_recall": 0.0,
            "max_precision": 0.0,
            "max_f1": 0.0,
            "max_recall_np": None,
            "max_precision_np": None,
            "max_f1_np": None,
        }
        
        # find max metrics
        clean_true_header = true_header_ts['clean_heading']
        if pred_greptext != None:
            for pred_np, pred_header_ts in pred_greptext.items():
                if clean_true_header in pred_header_ts['clean_heading']:
                    pred_textspan = pred_header_ts['clean_text']
                    pred_metric = eval_textspan(pred_textspan, true_header_ts['clean_text'])
                elif clean_true_header in pred_header_ts['clean_text']:
                    pred_textspan = pred_header_ts['clean_text']
                    pred_metric = eval_textspan(pred_textspan, true_header_ts['clean_text'])
                else:
                    continue
                if pred_metric['f1'] > metric_info['max_f1']:
                    metric_info['max_f1'] = pred_metric['f1']
                    metric_info['max_f1_np'] = pred_np
                if pred_metric['recall'] > metric_info['max_recall']:
                    metric_info['max_recall'] = pred_metric['recall']
                    metric_info['max_recall_np'] = pred_np
                if pred_metric['precision'] > metric_info['max_precision']:
                    metric_info['max_precision'] = pred_metric['precision']
                    metric_info['max_precision_np'] = pred_np
        # save metric_info for this np
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "a") as f:
            json.dump(metric_info, f)
            f.write("\n")
        
        max_recall_list.append(metric_info['max_recall'])
        max_precision_list.append(metric_info['max_precision'])
        max_f1_list.append(metric_info['max_f1'])

    avg_recall = sum(max_recall_list) / len(max_recall_list) if len(max_recall_list) > 0 else 0.0
    avg_precision = sum(max_precision_list) / len(max_precision_list) if len(max_precision_list) > 0 else 0.0
    avg_f1 = sum(max_f1_list) / len(max_f1_list) if len(max_f1_list) > 0 else 0.0
    
    return avg_recall, avg_precision, avg_f1


def eval_sht_per_dataset_top_down(dataset, sht_type):
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
    
    recall_list = []
    precision_list = []
    f1_list = []
    for file_id in range(start_id, end_id):
        file_name = str(file_id)
        avg_recall, avg_precision, avg_f1 = eval_sht_per_doc_topdown(dataset, file_name, sht_type)
        recall_list.append(avg_recall)
        precision_list.append(avg_precision)
        f1_list.append(avg_f1)

    avg_ds_recall = sum(recall_list) / len(recall_list)
    avg_ds_precision = sum(precision_list) / len(precision_list)
    avg_ds_f1 = sum(f1_list) / len(f1_list)

    return avg_ds_recall, avg_ds_precision, avg_ds_f1

if __name__ == "__main__":
    # sht_type_list = [
    #     'grobid',
    #     'llm_txt_sht',
    #     'llm_vision_sht',
    #     ''
    # ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sht_type", type=str, required=True, help="SHT type to evaluate")
    args = parser.parse_args()
    sht_type = args.sht_type


    recall_list = []
    precision_list = []
    f1_list = []

    for dataset in DATASET_LIST:
        avg_recall, avg_precision, avg_f1 = eval_sht_per_dataset_top_down(dataset, sht_type)
        recall_list.append(avg_recall)
        precision_list.append(avg_precision)
        f1_list.append(avg_f1)

    print_str = f"{sht_type},{','.join([f'{recall:.2f}' for recall in recall_list])},{','.join([f'{precision:.2f}' for precision in precision_list])},{','.join([f'{f1:.2f}' for f1 in f1_list])}"


    