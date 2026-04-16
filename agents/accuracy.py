import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
# set logging level as info
logging.getLogger().setLevel(logging.INFO)
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_result_path, get_simple_qasper_docs, extract_numbers, get_unanswerable_qasper_qids, get_notmentioned_contract_qids, get_hard_queries
from eval.utils import global_normalize_answer, get_gold_answers
from agents.office_reward import score_answer

INVALID_CIVIC_QUERY_IDS = [qid for qid in range(0, 380) if (qid % 20 >= 10)] + [82, 201, 264, 282, 300, 389, 398, 406, 263, 285] + (list(range(204, 210)) + list(range(305, 310)) + [401, 411])
INVALID_CIVIC_NEW_QUERY_IDS = []
INVALID_QASPER_DOCS = get_simple_qasper_docs()
INVALID_QASPER_QUERY_IDS = get_unanswerable_qasper_qids()
INVALID_CONTRACT_QUERY_IDS = get_notmentioned_contract_qids()

OFFICE_ACC_TOLERANCE = 0.00

# def is_correct(gt, llm_answer, dataset):
#     if dataset == 'civic':
#         assert isinstance(gt, str) or isinstance(gt, list)
#         if isinstance(gt, str):

def avg_llm_judge_score_per_dataset(dataset, method, model, sht_type, hard_q_only):
    result_path = get_result_path(dataset, model, method, sht_type)
    llm_judge_path = str(result_path).replace("/core/", "/llm_judge_response/")
    results = []
    hard_query_ids = None
    if hard_q_only == True:
        hard_query_ids = get_hard_queries(dataset)
    with open(llm_judge_path, 'r') as file:
        for l in file.readlines():
            result = json.loads(l)
            assert 'score' in result, f"Missing score in result: {result} ({llm_judge_path})"
            # if dataset == 'qasper' and result['id'] in INVALID_QASPER_QUERY_IDS:
            #     continue
            if hard_q_only == True and result['id'] not in hard_query_ids:
                continue
            results.append(result['score'])
    factor = 1 if dataset in ["finance", 'finance_rand', 'finance_rand_v1'] else 5


    num_results = len(results)
    failed_results = 0
    if hard_q_only == False:
        if dataset == 'finance' and method == 'react_agent_grep_next' and model == 'gpt-5.4' and sht_type == 'deep':
            failed_results = 3
        if dataset == 'finance' and method == 'react_agent_grep_next' and model == 'gpt-5.4' and sht_type == '':
            failed_results = 1
        if dataset == 'finance' and method == 'react_agent_grep_next_chunk_notoc' and model == 'gpt-5-mini' and sht_type == 'intrinsic':
            failed_results = 1
        if dataset == 'qasper' and method == 'react_agent_grep_next_chunk_notoc' and model == 'gpt-5-mini' and sht_type == 'intrinsic':
            failed_results = 7
        if dataset == 'finance' and method == 'react_agent' and model == 'gpt-5.4' and sht_type == "deep":
            failed_results = 3
        if dataset == 'finance' and method == 'react_agent' and model == 'gpt-5.4' and sht_type == "":
            failed_results = 1
        if dataset == 'finance_rand_v1' and method == 'react_agent_clean' and model == 'gpt-5.4' and sht_type == 'deep':
            failed_results = 100 - 36
        if dataset == 'finance_rand_v1' and method == 'react_agent_clean' and model == 'gpt-5.4' and sht_type == 'llm_txt_sht':
            failed_results = 5
        if dataset == 'qasper_rand_v1' and method == 'react_agent_clean' and model == 'gpt-5.4' and sht_type == 'llm_vision_sht':
            failed_results = 2
        if dataset == "qasper_rand_v1" and method == 'baseline' and model == 'gpt-5.4' and sht_type == 'intrinsic':
            failed_results = 2
    
    logging.info(f"{dataset} ({method}, {sht_type}): {num_results}+{failed_results} queries evaluated")

    return sum(results) / ((num_results + failed_results) * factor) if (num_results + failed_results) > 0 else None

            
def token_f1_score_per_answer(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = global_normalize_answer(prediction).split()
    ground_truth_tokens = global_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    # return precision

def token_f1_score_per_query(dataset, gt, llm_answer):
    if dataset in ['civic', 'civic_new', 'contract_rand_v0_1', 'civic_rand', 'civic_rand_v1']:
        if isinstance(gt, str):
            # if gt is in the format of yyyy-mm, we can convert it into yyyy, and use words to replace mm
            candid_gts = []
            if re.match(r'^\d{4}-\d{2}$', gt):
                year, month = gt.split("-")
                if month.isdigit() and 1 <= int(month) <= 12:
                    month_word = month_name[int(month)]
                    candid_gts.append(f"{year} {month_word}")
                    candid_gts.append(f"{year} {int(month)}")
                    # abbreviated month name
                    month_abbr = month_word[:3]
                    candid_gts.append(f"{year} {month_abbr}")
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', gt):
                year, month, day = gt.split("-")
                if month.isdigit() and 1 <= int(month) <= 12 and day.isdigit() and 1 <= int(day) <= 31:
                    month_word = month_name[int(month)]
                    candid_gts.append(f"{year} {month_word} {day}")
                    candid_gts.append(f"{year} {int(month)} {int(day)}")
                    # abbreviated month name
                    month_abbr = month_word[:3]
                    candid_gts.append(f"{year} {month_abbr} {day}")
            # split date strings: replace all non-alphanumeric characters with spaces, then split by whitespace
            gt = re.sub(r'[^A-Za-z0-9]+', ' ', gt)
            candid_gts.append(gt)
            llm_answer = re.sub(r'[^A-Za-z0-9]+', ' ', llm_answer)
            return max([token_f1_score_per_answer(llm_answer, cgt) for cgt in candid_gts])
        elif isinstance(gt, list):
            if len(gt) == 0:
                gt_str = "none"
            else:
                gt_str = "\n".join(gt)
            return token_f1_score_per_answer(llm_answer, gt_str)
    elif dataset == "qasper":
        if isinstance(gt, str):
            return token_f1_score_per_answer(llm_answer, gt)
        elif isinstance(gt, list):
            if len(gt) == 0:
                gt_str = "Unanswerable"
                return token_f1_score_per_answer(llm_answer, gt_str)
            # return sum([token_f1_score_per_answer(llm_answer, gt_i) for gt_i in gt]) / len(gt)
            return max([token_f1_score_per_answer(llm_answer, gt_i) for gt_i in gt])
    elif dataset == "finance":
        assert len(gt) == 1
        llm_answer = llm_answer.replace(",", "")
        gt[0] = gt[0].replace(",", "")
        candid_gts = [gt[0]]
        candid_llm_answers = [llm_answer]
        if len(re.findall(r'-?\d+\.?\d*', gt[0])) == 1:
            # numeric reasoning query
            candid_gts += extract_numbers(gt[0])
            candid_llm_answers += extract_numbers(llm_answer)
        if gt[0].lower().startswith("yes.") or gt[0].lower().startswith("no."):
            candid_gts.append("yes" if gt[0].lower().startswith("yes.") else "no")
            if "yes" in llm_answer.lower() and "no" not in llm_answer.lower():
                candid_llm_answers.append("yes")
            elif "no" in llm_answer.lower() and "yes" not in llm_answer.lower():
                candid_llm_answers.append("no")
        return max([token_f1_score_per_answer(llm_a, gt_a) for llm_a in candid_llm_answers for gt_a in candid_gts])
        # return token_f1_score_per_answer(llm_answer, gt[0])
    elif dataset in ["contract", "contract_new", 'contract_rand', 'contract_rand_v1', 'contract_rand_v2', 'contract_rand_v3']:
        assert isinstance(gt, str)
        return token_f1_score_per_answer(llm_answer, gt)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def accuracy_per_dataset(dataset, method, model, sht_type, hard_q_only):
    result_path = get_result_path(dataset, model, method, sht_type)

    with open(result_path, 'r') as file:
        results = sorted([json.loads(line) for line in file.readlines()], key=lambda x: x['id'])

    gold_answers = get_gold_answers(dataset)

    if dataset == 'civic':
        results = [r for r in results if r['id'] not in INVALID_CIVIC_QUERY_IDS]
        gold_answers = [a for a in gold_answers if a['id'] not in INVALID_CIVIC_QUERY_IDS]

    token_f1_scores = []

    hard_query_ids = None
    if hard_q_only == True:
        hard_query_ids = get_hard_queries(dataset)

    for result, gold_answer in zip(results, gold_answers):
        # if dataset == "contract":
        #     if gold_answer['answer'].lower() == "notmentioned":
        #         continue
        # if dataset == "contract":
        #     if result['id'] in INVALID_CONTRACT_QUERY_IDS:
        #         continue
        assert result['id'] == int(gold_answer['id']), f"ID mismatch: {result['id']} vs {gold_answer['id']}"
        if hard_q_only == True and result['id'] not in hard_query_ids:
            continue
        # if dataset == "qasper":
        #     if gold_answer['file_name'] in INVALID_QASPER_DOCS:
        #         continue
        # if dataset == "finance":
        #     if not len(re.findall(r'-?\d+\.?\d*', gold_answer['answer'][0])) == 1:
        #         continue
        if result['is_success'] == False:
            logging.warning(f"query {result['id']} failed!\n\t{result['message']}")
            token_f1_scores.append(0.0)
        else:
            assert result['id'] == int(gold_answer['id'])
            token_f1 = token_f1_score_per_query(dataset, gold_answer['answer'], result['message'])
            # print(f"query id {result['id']}, token f1: {token_f1} (pred: {result['message']}, gt: {gold_answer['answer']})")
            token_f1_scores.append(token_f1)

    if hard_q_only == False:
        assert len(token_f1_scores) == len(results)
    
    logging.info(f"{dataset} ({method}, {sht_type}): {len(token_f1_scores)} queries evaluated")
    # if dataset != 'qasper':
    #     assert len(token_f1_scores) == len(results)
    # else:
    #     assert len(token_f1_scores) == len(results) - len(INVALID_QASPER_QUERY_IDS)
    # if dataset in ['contract', 'qasper', 'finance']:
    #     assert len(token_f1_scores) == len(results)
    # elif dataset == "civic":
    #     assert len(token_f1_scores) == len(results) - len(INVALID_CIVIC_QUERY_IDS)
    # elif dataset == "qasper":
    #     print(f"qasper query number: {len(token_f1_scores)}")
    # elif dataset == "contract":
        # print(f"contract query number: {len(token_f1_scores)}")
    # elif dataset == "finance":
    #     print(f"finance query number: {len(token_f1_scores)}")

    
    avg_token_f1 = sum(token_f1_scores) / len(token_f1_scores) if len(token_f1_scores) > 0 else None

    return avg_token_f1

def avg_score_office(method, model, sht_type):
    print(f"Evaluating office dataset with method {method}, model {model}, sht_type {sht_type}...")
    result_path = get_result_path(dataset, model, method, sht_type)

    with open(result_path, 'r') as file:
        results = sorted([json.loads(line) for line in file.readlines()], key=lambda x: x['id'])

    gold_answers = get_gold_answers(dataset)

    score_list = []
    for result, gold_answer in zip(results, gold_answers):
        assert result['id'] == int(gold_answer['id']), f"ID mismatch: {result['id']} vs {gold_answer['id']}"
        if result['is_success'] == False:
            logging.warning(f"[office, {method}, {model}, {sht_type}] query {result['id']} failed!\n\t{result['message']}")
            score = 0.0
        else:
            score = score_answer(gold_answer['answer'], result['message'], tolerance=OFFICE_ACC_TOLERANCE)
            # print(f"query id {result['id']}, score: {score} (pred: {result['message']}, gt: {gold_answer['answer']})")
        score_list.append(score)
        # print(f"office query id {result['id']}, score: {score} (pred: {result['message']}, gt: {gold_answer['answer']})")
    
    assert len(score_list) == len(results)
    avg_score = sum(score_list) / len(score_list) if len(score_list) > 0 else None
    logging.info(f"office ({method}, {sht_type}): {len(score_list)} queries evaluated")
    return avg_score
        

if __name__ == "__main__":
    HARD_Q_ONLY = False
    model = "gpt-5.4"
    # model = "gpt-5-mini"
    method_list = [
        # "baseline",
        # "toc_in_context",
        # "react_agent_grep_next_chunk_notoc",
        "react_agent_clean",
        # 'react_agent_grep_next_chunk_clean',
    ]
    sht_type_list = [
        'deep',
        'wide',
        'grobid',
        # 'intrinsic',
        'llm_txt_sht',
        'llm_vision_sht',
        '',
    ]
    # print results as a pandas table
    # column = dataset, row = method and sht_type, cell = accuracy
    results = []
    for dataset in DATASET_LIST:
    # for dataset in ['qasper_rand_v1']:
        for sht_type in sht_type_list:
            for method in method_list:
                if dataset in ['contract_rand_v0_1', 'civic_rand_v1']:
                    avg_score = accuracy_per_dataset(dataset, method, model, sht_type, HARD_Q_ONLY)
                else:
                    avg_score = avg_llm_judge_score_per_dataset(dataset, method, model, sht_type, HARD_Q_ONLY)
                results.append({
                    "dataset": dataset,
                    "method": method,
                    'sht_type': sht_type,
                    "avg_score": avg_score
                })
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=["method", "sht_type"],
        columns="dataset",
        values="avg_score"
    ).reset_index()

    # Ensure column order
    desired_columns = ["method", "sht_type"] + DATASET_LIST
    # , "civic", "contract", "finance", "qasper", 'civic_new', 'office', 'contract_new', 'contract_rand', 'contract_rand_v1', 'contract_rand_v2', 'contract_rand_v3', 'contract_rand_v0_1', 'civic_rand', 'civic_rand_v1', 'finance_rand', 'finance_rand_v1', 'qasper_rand_v1']
    pivot_df = pivot_df.reindex(columns=desired_columns)

    # ---- Sorting ----
    # 1. Method order
    # method_order = ["toc_in_context", "react_agent_grep_next"]
    method_order = method_list
    pivot_df["method"] = pd.Categorical(
        pivot_df["method"], categories=method_order, ordered=True
    )

    # 2. sht_type order
    sht_order = ["deep", "wide", "grobid", 'llm_txt_sht', 'llm_vision_sht', "",  "intrinsic"]
    pivot_df["sht_type"] = pd.Categorical(
        pivot_df["sht_type"], categories=sht_order, ordered=True
    )

    # Sort
    pivot_df = pivot_df.sort_values(by=["method", "sht_type"])

    # ---- Renaming sht_type ----
    pivot_df["sht_type"] = pivot_df["sht_type"].replace({
        "": "shed",
        "intrinsic": "true"
    })

    # Print
    print(pivot_df.round(2).to_csv(index=False))

    

    # # print(df.pivot(index='method', columns='dataset', values='avg_token_f1'))
    # # print rows in order: baseline, toc_in_context, react_agent
    # df['method'] = pd.Categorical(df['method'], categories=method_list, ordered=True)
    # df = df.sort_values('method')
    # print(df.pivot(index='method', columns='dataset', values='avg_score').round(4).to_csv())



    