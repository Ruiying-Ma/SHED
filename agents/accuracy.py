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
from agents.utils import get_result_path, get_simple_qasper_docs, extract_numbers, get_unanswerable_qasper_qids, get_notmentioned_contract_qids
from eval.utils import global_normalize_answer, get_gold_answers

INVALID_CIVIC_QUERY_IDS = [qid for qid in range(0, 380) if (qid % 20 >= 10)] + [82, 201, 264, 282, 300, 389, 398, 406, 263, 285] + (list(range(204, 210)) + list(range(305, 310)) + [401, 411])
INVALID_QASPER_DOCS = get_simple_qasper_docs()
INVALID_QASPER_QUERY_IDS = get_unanswerable_qasper_qids()
INVALID_CONTRACT_QUERY_IDS = get_notmentioned_contract_qids()

# def is_correct(gt, llm_answer, dataset):
#     if dataset == 'civic':
#         assert isinstance(gt, str) or isinstance(gt, list)
#         if isinstance(gt, str):

def avg_llm_judge_score_per_dataset(dataset, method, model):
    result_path = get_result_path(dataset, model, method)
    llm_judge_path = str(result_path).replace("/core/", "/llm_judge_response/")
    results = []
    with open(llm_judge_path, 'r') as file:
        for l in file.readlines():
            result = json.loads(l)
            assert 'score' in result, f"Missing score in result: {result} ({llm_judge_path})"
            # if dataset == 'qasper' and result['id'] in INVALID_QASPER_QUERY_IDS:
            #     continue
            results.append(result['score'])
    factor = 1 if dataset == "finance" else 5
    logging.info(f"{dataset}: {len(results)} queries evaluated")
    return sum(results) / (len(results) * factor)

            
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
    if dataset == 'civic':
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
    elif dataset == "contract":
        assert isinstance(gt, str)
        return token_f1_score_per_answer(llm_answer, gt)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def accuracy_per_dataset(dataset, method, model):
    result_path = get_result_path(dataset, model, method)

    with open(result_path, 'r') as file:
        results = sorted([json.loads(line) for line in file.readlines()], key=lambda x: x['id'])

    gold_answers = get_gold_answers(dataset)

    if dataset == 'civic':
        results = [r for r in results if r['id'] not in INVALID_CIVIC_QUERY_IDS]
        gold_answers = [a for a in gold_answers if a['id'] not in INVALID_CIVIC_QUERY_IDS]

    token_f1_scores = []

    for result, gold_answer in zip(results, gold_answers):
        # if dataset == "contract":
        #     if gold_answer['answer'].lower() == "notmentioned":
        #         continue
        # if dataset == "contract":
        #     if result['id'] in INVALID_CONTRACT_QUERY_IDS:
        #         continue
        assert result['id'] == int(gold_answer['id']), f"ID mismatch: {result['id']} vs {gold_answer['id']}"
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

    assert len(token_f1_scores) == len(results)
    logging.info(f"{dataset}: {len(token_f1_scores)} queries evaluated")
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

    
    avg_token_f1 = sum(token_f1_scores) / len(token_f1_scores)

    return avg_token_f1

if __name__ == "__main__":
    # model = "gpt-5.4"
    model = "gpt-5-mini"
    method_list = [
        "baseline",
        "toc_in_context",
        "react_agent",
        "react_agent_grep_all",
        "react_agent_grep_id",
        "react_agent_grep_next",
        "react_agent_grep_next_notoc"
    ]
    # print results as a pandas table
    # column = dataset, row = method, cell = accuracy
    results = []
    for dataset in DATASET_LIST:
        for method in method_list:
            if dataset in ['civic', 'contract']:
                avg_score = accuracy_per_dataset(dataset, method, model)
            else:
                avg_score = avg_llm_judge_score_per_dataset(dataset, method, model)
            results.append({
                "dataset": dataset,
                "method": method,
                "avg_score": avg_score
            })
    df = pd.DataFrame(results)
    # print(df.pivot(index='method', columns='dataset', values='avg_token_f1'))
    # print rows in order: baseline, toc_in_context, react_agent
    df['method'] = pd.Categorical(df['method'], categories=method_list, ordered=True)
    df = df.sort_values('method')
    print(df.pivot(index='method', columns='dataset', values='avg_score').round(4).to_csv())



    