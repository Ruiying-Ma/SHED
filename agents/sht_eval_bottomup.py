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
from agents.merge_toc_textspan_clean import merge_toc_textspan_per_query

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
    

def load_textspan(dataset, qinfo, sht_type):
    file_name = qinfo['file_name']
    # if sht_type == "deep":
    #     try:
    #         m_np_ts = merge_toc_textspan_per_query(
    #             orig_dataset=dataset.split("_")[0],
    #             new_file_name=qinfo["file_name"],
    #             sht_type="deep",
    #             new_file_names=qinfo['new_file_names'],
    #             merged_dataset="finance_rand_v1",
    #             need_store=False,
    #         )
    #     except Exception as e:
    #         logging.warning(f"Failed to merge toc textspan for {dataset}/{qinfo['file_name']} (deep), error: {e}, traceback: {traceback.format_exc()}")
    #         return None
    # else:
        # textspan_clean_path = os.path.join(
        #     DATA_ROOT_FOLDER,
        #     dataset,
        #     sht_type,
        #     "toc_textspan_clean",
        #     file_name + ".json"
        # )
        # if not os.path.exists(textspan_clean_path):
        #     logging.warning(f"Textspan clean not found for {dataset}/{file_name} ({sht_type}), path: {textspan_clean_path}")
        #     return None
        # with open(textspan_clean_path, "r") as f:
        #     m_np_ts = json.load(f)
    
    if sht_type in ['deep']:
        grep_text_sht_type = ''
    else:
        grep_text_sht_type = sht_type
    greptext_clean_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        grep_text_sht_type,
        "toc_greptext_clean",
        file_name + ".json"
    )
    if not os.path.exists(greptext_clean_path):
        logging.warning(f"Greptext clean not found for {dataset}/{file_name} ({sht_type}), path: {greptext_clean_path}")
        return None
    
    with open(greptext_clean_path, "r") as f:
        m_np_header = json.load(f)

    # assert len(m_np_ts) == len(m_np_header), f"Number of NPs in textspan and greptext do not match for {dataset}/{file_name} ({sht_type})"

    if sht_type == 'deep':
        # sort np keys in m_np_header (as tuple)
        m_np_header = dict(sorted(m_np_header.items(), key=lambda x: tuple([int(i) for i in x[0].split('.')])))
        # assign 1.1.1....
        new_m_np_header = dict()
        for i, (np, hd) in enumerate(m_np_header.items()):
            new_np = ".".join(["1" for _ in range(i+1)])
            new_m_np_header[new_np] = hd
        m_np_header = new_m_np_header
    
    # assert set(m_np_ts.keys()) == set(m_np_header.keys()), f"NP keys in textspan and greptext do not match for {dataset}/{file_name} ({sht_type})\n{set(m_np_ts.keys())}\n{set(m_np_header.keys())}"
    # clean header and textspan
    m_np_header_ts = dict()
    for np in sorted(m_np_header.keys(), key=lambda x: tuple([int(i) for i in x.split('.')])):
        if sht_type != 'llm_txt_sht' and sht_type != 'llm_vision_sht':
            if np == "1":
                continue
        header = m_np_header[np]['heading']
        greptext = m_np_header[np]['text']
        m_np_header_ts[np] = dict()
        m_np_header_ts[np]['heading'] = header
        m_np_header_ts[np]['clean_heading'] = clean_txt(header)
        m_np_header_ts[np]['clean_text'] = clean_txt(greptext) # excluding heading
        # get clean text of all ancestors, including the heading itself
        ancestor_np_list = []
        level_id_list = np.split('.') 
        start_id = 1
        if sht_type != 'llm_txt_sht' and sht_type != 'llm_vision_sht':
            assert len(level_id_list) > 1
            assert level_id_list[0] == "1", f"First level id should be 1 for non-LLM SHT, but got {level_id_list[0]} for np {np} in {dataset}/{file_name} ({sht_type})"
            start_id = 2
        for i in range(start_id, len(level_id_list)+1):
            ancestor_np_list.append('.'.join(level_id_list[:i]))
        
        m_np_header_ts[np]['clean_ancestor_text'] = " ".join(
            [m_np_header_ts[anc_np]['clean_heading'].strip()
             for anc_np in ancestor_np_list # anc_np must be in m_np_header_ts
            ]
        )

    return m_np_header_ts
    
def eval_sht_per_doc_bottom_up(dataset, qinfo, sht_type):
    file_name = qinfo['file_name']
    print(f"Evaluating {dataset}/{file_name} ({sht_type})")
    true_greptext = load_textspan(dataset, qinfo, "intrinsic")
    pred_greptext = load_textspan(dataset, qinfo, sht_type)

    if len(true_greptext) == 0:
        return None
    

    if pred_greptext is None:
        logging.warning(f"No predicted greptext found for {dataset}/{file_name} ({sht_type}), skipping evaluation")
        return 0, 0.0, 0.0


    dst_path = Path(__file__).resolve().parent / "results" / "sht_eval" / "bottom_up" / sht_type / dataset / (file_name + ".jsonl")
    
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
                    pred_metric = eval_textspan(pred_header_ts['clean_ancestor_text'], true_header_ts['clean_ancestor_text'])
                elif clean_true_header in pred_header_ts['clean_text']:
                    pred_metric = eval_textspan(pred_header_ts['clean_ancestor_text'], true_header_ts['clean_ancestor_text'])
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


def eval_sht_per_dataset_bottom_up(dataset, sht_type):
    if dataset == "civic_rand_v1":
        start_id = 0
        end_id = 107
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
        raise ValueError(f"Invalid dataset: {dataset}")
    
    recall_list = []
    precision_list = []
    f1_list = []
    with open(Path(DATA_ROOT_FOLDER) / dataset / "queries.json", "r") as f:
        queries = json.load(f)

    
    for qinfo in queries[start_id:end_id]:
        avg_recall, avg_precision, avg_f1 = eval_sht_per_doc_bottom_up(dataset, qinfo, sht_type)
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

    for dataset in DATASET_LIST[2:]:
        avg_recall, avg_precision, avg_f1 = eval_sht_per_dataset_bottom_up(dataset, sht_type)
        recall_list.append(avg_recall)
        precision_list.append(avg_precision)
        f1_list.append(avg_f1)

    print_str = f"{sht_type},{','.join([f'{recall:.2f}' for recall in recall_list])},{','.join([f'{precision:.2f}' for precision in precision_list])},{','.join([f'{f1:.2f}' for f1 in f1_list])}"


    