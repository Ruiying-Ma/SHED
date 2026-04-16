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


def detail_latency_per_dataset(dataset, model, method, sht_type):
    result_path = get_result_path(dataset, model, method, sht_type)
    
    latency_list = []

    with open(result_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            if dataset == "civic":
                if result['id'] in INVALID_CIVIC_QUERY_IDS:
                    continue

            latency_list.append(result['latency'])

    return latency_list

def avg_latency_ratio(dataset, method_1, method_2, model, sht_type):
    """Compare the average input token usage ratio: method_1 / method_2"""
    latency_list_1 = detail_latency_per_dataset(dataset, model, method_1, sht_type)
    latency_list_2 = detail_latency_per_dataset(dataset, model, method_2, sht_type)

    assert len(latency_list_1) == len(latency_list_2), f"{dataset}, {len(latency_list_1)}, {len(latency_list_2)}"
    ratios = [i / j for i, j in zip(latency_list_1, latency_list_2)]
    avg_ratio = sum(ratios) / len(ratios)
    logging.info(f"{dataset} ({method}, {sht_type}): {len(ratios)} queries evaluated")
    return avg_ratio

def avg_latency(dataset, method_1, model, sht_type):
    latency_list_1 = detail_latency_per_dataset(dataset, model, method_1, sht_type)

    avg_latency = sum(latency_list_1) / len(latency_list_1)

    logging.info(f"{dataset} ({method}, {sht_type}): {len(latency_list_1)} queries evaluated")
    return avg_latency


if __name__ == "__main__":
    # print ratio as a pandas dataframe
    # row = method, baseline is all 1
    # column = dataset
    # model = "gpt-5-mini"
    model = "gpt-5.4"
    methods = [
        # "baseline",
        # "toc_in_context",
        # "react_agent_grep_next_chunk_notoc",
        "react_agent_clean",
        # 'react_agent_grep_next_chunk_clean',
    ]
    sht_types = [
        'deep', 
        'wide', 
        'grobid', 
        '', 
        'llm_txt_sht',
        'intrinsic'

    ]
    results = []
    for sht_type in sht_types:
        for dataset in DATASET_LIST:
            for method in methods:
                # avg_ratio = avg_latency_ratio(dataset, method, 'baseline', model, sht_type)
                # avg_ratio = avg_output_token_usage_ratio(dataset, method, 'baseline', model)
                # avg_ratio = avg_input_cost_ratio(dataset, method, 'baseline', model)
                avg_ratio = avg_latency(dataset, method, model, sht_type)
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
    pivot_df = pivot_df.reindex(columns=desired_columns)

    # ---- Sorting ----
    # 1. Method order
    method_order = methods
    pivot_df["method"] = pd.Categorical(
        pivot_df["method"], categories=method_order, ordered=True
    )

    # 2. sht_type order
    sht_order = ["deep", "wide", "grobid", 'llm_txt_sht', 'llm_vision_sht', "", "intrinsic"]
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
    print(pivot_df.round(1).to_csv(index=False))

    # df = pd.DataFrame(results)

    # df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    # df = df.sort_values('method')
    # # print(df.to_csv(index=False))
    # # print as csv
    # print(df.pivot(index='method', columns='dataset', values='avg_ratio').round(4).to_csv())


