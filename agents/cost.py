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


def detail_tokens_per_dataset(dataset, model, method, sht_type):
    result_path = get_result_path(dataset, model, method, sht_type)
    
    input_tokens = []
    cached_tokens = []
    output_tokens = []


    with open(result_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if dataset == "civic":
                if result['id'] in INVALID_CIVIC_QUERY_IDS:
                    continue
            input_tokens.append(result['input_tokens'])
            cached_tokens.append(result['cached_tokens'])
            output_tokens.append(result['output_tokens'])
    
    return input_tokens, cached_tokens, output_tokens



def avg_input_token_usage_ratio(dataset, method_1, method_2, model, sht_type):
    """Compare the average input token usage ratio: method_1 / method_2"""
    input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1, sht_type)
    input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2, sht_type)

    ratios = []
    assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2), f"{dataset}, {len(input_tokens_1)}, {len(input_tokens_2)}"


    for i in range(len(input_tokens_1)):
        total_tokens_1 = input_tokens_1[i] + cached_tokens_1[i]
        total_tokens_2 = input_tokens_2[i] + cached_tokens_2[i]

        # if dataset == "contract" and total_tokens_2 < 5000:
        #     continue

        ratios.append(total_tokens_1 / total_tokens_2)

    logging.info(f"{dataset} ({method}, {sht_type}): {len(ratios)} queries evaluated.")
    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio

def total_input_cost(dataset, method, model, sht_type):
    """Calculate the average input token usage ratio: method_1 / method_2"""
    def get_cost(mode, input, cached, method):
        if method in ["baseline", "toc_in_context"]:
            return get_cost_usd(mode, input + cached, 0, 0)
        else:
            return get_cost_usd(mode, input, cached, 0)

    input_tokens, cached_tokens, _ = detail_tokens_per_dataset(dataset, model, method, sht_type)

    total_cost = 0
    for i in range(len(input_tokens)):
        total_cost += get_cost(model, input_tokens[i], cached_tokens[i], method)

    logging.info(f"{dataset} ({method}, {sht_type}): {len(input_tokens)} queries evaluated.")
    return total_cost


def amortized_tree_gen_cost(dataset, method, model, sht_type):
    doc_gen_path = str(get_result_path(dataset, model, "baseline", sht_type)).replace("/core/", "/llm_gen_toc_response/").replace("baseline", sht_type.replace("_sht", ""))
    m_file_cost = dict()
    with open(doc_gen_path, 'r') as f:
        for line in f:
            doc_gen_result = json.loads(line)
            filename = doc_gen_result['file_name']
            input_tokens = doc_gen_result['input_tokens']
            cached_tokens = doc_gen_result['cached_tokens']
            output_tokens = doc_gen_result['output_tokens']
            m_file_cost[filename] = get_cost_usd(model, input_tokens + cached_tokens, 0, output_tokens)
    
    result_path = get_result_path(dataset, model, method, sht_type)
    qids = []
    with open(result_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if dataset == "civic":
                if result['id'] in INVALID_CIVIC_QUERY_IDS:
                    continue
            qids.append(result['id'])
    
    queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    m_qid_filename = {qinfo['id']: qinfo['file_name'] for qinfo in queries}
    m_filename_cnt = dict()
    for qid in qids:
        filename = m_qid_filename[qid]
        if filename not in m_filename_cnt:
            m_filename_cnt[filename] = 0
        m_filename_cnt[filename] += 1
    
    m_filename_amort_cost = dict()
    for filename, cnt in m_filename_cnt.items():
        gen_cost = m_file_cost[filename]
        amort_cost = gen_cost / cnt
        m_filename_amort_cost[filename] = amort_cost

    file_gen_cost_per_query_amort = [m_filename_amort_cost[m_qid_filename[qid]] for qid in qids]
    logging.info(f"{dataset} ({method}, {sht_type}): {len(file_gen_cost_per_query_amort)} queries evaluated.")
    return sum(file_gen_cost_per_query_amort) / len(file_gen_cost_per_query_amort)



    
def total_tree_gen_cost(dataset, method, model, sht_type):
    doc_gen_path = str(get_result_path(dataset, model, "baseline", sht_type)).replace("/core/", "/llm_gen_toc_response/").replace("baseline", sht_type.replace("_sht", ""))
    print(doc_gen_path)
    m_file_cost = dict()
    with open(doc_gen_path, 'r') as f:
        for line in f:
            doc_gen_result = json.loads(line)
            filename = doc_gen_result['file_name']
            input_tokens = doc_gen_result['input_tokens']
            cached_tokens = doc_gen_result['cached_tokens']
            output_tokens = doc_gen_result['output_tokens']
            m_file_cost[filename] = get_cost_usd(model, input_tokens + cached_tokens, 0, output_tokens)
    
    result_path = get_result_path(dataset, model, method, sht_type)
    qids = []
    with open(result_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if dataset == "civic":
                if result['id'] in INVALID_CIVIC_QUERY_IDS:
                    continue
            qids.append(result['id'])
    
    queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    m_qid_filename = {qinfo['id']: qinfo['file_name'] for qinfo in queries}

    file_gen_cost_sum = 0
    calculated_files = set()
    for qid in qids:
        filename = m_qid_filename[qid]
        if filename in calculated_files:
            continue
        file_gen_cost_sum += m_file_cost[filename]
        calculated_files.add(filename)
    logging.info(f"{dataset} ({method}, {sht_type}): {len(qids)} queries evaluated.")
    return file_gen_cost_sum



def avg_input_cost_ratio(dataset, method_1, method_2, model, sht_type):
    """Compare the average input token usage ratio: method_1 / method_2"""
    def get_cost(mode, input, cached, method):
        if method in ["baseline", "toc_in_context"]:
            return get_cost_usd(mode, input + cached, 0, 0)
        else:
            return get_cost_usd(mode, input, cached, 0)

    input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1, sht_type)
    input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2, sht_type)

    ratios = []
    assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2), f"{dataset}, {len(input_tokens_1)}, {len(input_tokens_2)}"

    for i in range(len(input_tokens_1)):
        total_cost_1 = get_cost(model, input_tokens_1[i], cached_tokens_1[i], method_1)
        total_cost_2 = get_cost(model, input_tokens_2[i], cached_tokens_2[i], method_2)

        ratios.append(total_cost_1 / total_cost_2)

    logging.info(f"{dataset} ({method}, {sht_type}): {len(ratios)} queries evaluated.")

    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio


# def plot_input_token_usage_ratio_vs_baseline_per_dataset(dataset, method_1, method_2, model):
#     input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1)
#     input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2)

#     ratios = []
#     assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2)
#     for i in range(len(input_tokens_1)):
#         total_tokens_1 = input_tokens_1[i] + cached_tokens_1[i]
#         total_tokens_2 = input_tokens_2[i] + cached_tokens_2[i]

#         ratios.append(total_tokens_1 / total_tokens_2)

#     print(min([i + c for i, c in zip(input_tokens_2, cached_tokens_2)]))

#     plt.plot([i + c for i, c in zip(input_tokens_2, cached_tokens_2)], ratios, 'o')
#     plt.show()


def avg_output_token_usage_ratio(dataset, method_1, method_2, model, sht_type):
    """Compare the average output token usage ratio: method_1 / method_2"""
    _, _, output_tokens_1 = detail_tokens_per_dataset(dataset, model, method_1, sht_type)
    _, _, output_tokens_2 = detail_tokens_per_dataset(dataset, model, method_2, sht_type)

    ratios = []
    assert len(output_tokens_1) == len(output_tokens_2)

    for i in range(len(output_tokens_1)):
        total_tokens_1 = output_tokens_1[i] 
        total_tokens_2 = output_tokens_2[i] 

        ratios.append(total_tokens_1 / total_tokens_2)
    
    logging.info(f"{dataset} ({method}, {sht_type}): {len(ratios)} queries evaluated.")
    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio


if __name__ == "__main__":
    # print ratio as a pandas dataframe
    # row = method, baseline is all 1
    # column = dataset
    # model = "gpt-5-mini"
    model = "gpt-5.4"
    methods = [
        # "baseline",
        # "react_agent_grep_next_chunk_notoc",
        # "toc_in_context",
        "react_agent_clean",
        # 'react_agent_grep_next_chunk_clean',
    ]
    sht_types = [
        'deep', 
        'wide', 
        'grobid', 
        'llm_txt_sht',
        'llm_vision_sht',
        '', 
        'intrinsic'

    ]
    results = []
    for dataset in DATASET_LIST:
    # for dataset in ['qasper_rand_v1']:
    # for dataset in ['office']:
        for sht_type in sht_types:
            for method in methods:
                # avg_ratio = avg_input_token_usage_ratio(dataset, method, 'baseline', model)
                # avg_ratio = avg_output_token_usage_ratio(dataset, method, 'baseline', model)
                # avg_ratio = avg_input_cost_ratio(dataset, method, 'baseline', model, sht_type)
                avg_ratio = total_input_cost(dataset, method, model, sht_type)
                # avg_ratio = amortized_tree_gen_cost(dataset, method, model, sht_type)
                # avg_ratio = total_tree_gen_cost(dataset, method, model, sht_type)
                results.append({
                    'dataset': dataset,
                    'method': method,
                    'sht_type': sht_type,
                    'avg_ratio': avg_ratio
                })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pivot to wide format
    pivot_df = df.pivot_table(
        index=["method", "sht_type"],
        columns="dataset",
        values="avg_ratio"
    ).reset_index()

    # Ensure column order
    desired_columns = ["method", "sht_type"] + DATASET_LIST
    #  "civic", "contract", "finance", "qasper", 'civic_new', 'office', 'contract_new', 'contract_rand', 'contract_rand_v1', "contract_rand_v2", 'qasper_rand_v1']
    pivot_df = pivot_df.reindex(columns=desired_columns)

    # ---- Sorting ----
    # 1. Method order
    # method_order = ["toc_in_context", "react_agent_grep_next"]
    method_order = methods
    pivot_df["method"] = pd.Categorical(
        pivot_df["method"], categories=method_order, ordered=True
    )

    # 2. sht_type order
    sht_order = sht_types
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





    # df = pd.DataFrame(results)

    # df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    # df = df.sort_values('method')
    # # print(df.to_csv(index=False))
    # # print as csv
    # print(df.pivot(index='method', columns='dataset', values='avg_ratio').round(4).to_csv())


