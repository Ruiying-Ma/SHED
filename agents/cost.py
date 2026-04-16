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


def detail_tokens_per_dataset(dataset, model, method):
    result_path = get_result_path(dataset, model, method)
    
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

def avg_input_token_usage_ratio(dataset, method_1, method_2, model):
    """Compare the average input token usage ratio: method_1 / method_2"""
    input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1)
    input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2)

    ratios = []
    assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2), f"{dataset}, {len(input_tokens_1)}, {len(input_tokens_2)}"


    for i in range(len(input_tokens_1)):
        total_tokens_1 = input_tokens_1[i] + cached_tokens_1[i]
        total_tokens_2 = input_tokens_2[i] + cached_tokens_2[i]

        # if dataset == "contract" and total_tokens_2 < 5000:
        #     continue

        ratios.append(total_tokens_1 / total_tokens_2)

    logging.info(f"{dataset}: {len(ratios)} queries evaluated.")
    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio

def avg_input_cost_ratio(dataset, method_1, method_2, model):
    """Compare the average input token usage ratio: method_1 / method_2"""
    def get_cost(mode, input, cached, method):
        if method == "baseline":
            return get_cost_usd(mode, input + cached, 0, 0)
        else:
            return get_cost_usd(mode, input, cached, 0)

    input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1)
    input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2)

    ratios = []
    assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2), f"{dataset}, {len(input_tokens_1)}, {len(input_tokens_2)}"

    for i in range(len(input_tokens_1)):
        total_cost_1 = get_cost(model, input_tokens_1[i], cached_tokens_1[i], method_1)
        total_cost_2 = get_cost(model, input_tokens_2[i], cached_tokens_2[i], method_2)

        ratios.append(total_cost_1 / total_cost_2)

    logging.info(f"{dataset}: {len(ratios)} queries evaluated.")

    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio


def plot_input_token_usage_ratio_vs_baseline_per_dataset(dataset, method_1, method_2, model):
    input_tokens_1, cached_tokens_1, _ = detail_tokens_per_dataset(dataset, model, method_1)
    input_tokens_2, cached_tokens_2, _ = detail_tokens_per_dataset(dataset, model, method_2)

    ratios = []
    assert len(input_tokens_1) == len(input_tokens_2) == len(cached_tokens_1) == len(cached_tokens_2)
    for i in range(len(input_tokens_1)):
        total_tokens_1 = input_tokens_1[i] + cached_tokens_1[i]
        total_tokens_2 = input_tokens_2[i] + cached_tokens_2[i]

        ratios.append(total_tokens_1 / total_tokens_2)

    print(min([i + c for i, c in zip(input_tokens_2, cached_tokens_2)]))

    plt.plot([i + c for i, c in zip(input_tokens_2, cached_tokens_2)], ratios, 'o')
    plt.show()


def avg_output_token_usage_ratio(dataset, method_1, method_2, model):
    """Compare the average output token usage ratio: method_1 / method_2"""
    _, _, output_tokens_1 = detail_tokens_per_dataset(dataset, model, method_1)
    _, _, output_tokens_2 = detail_tokens_per_dataset(dataset, model, method_2)

    ratios = []
    assert len(output_tokens_1) == len(output_tokens_2)

    for i in range(len(output_tokens_1)):
        total_tokens_1 = output_tokens_1[i] 
        total_tokens_2 = output_tokens_2[i] 

        ratios.append(total_tokens_1 / total_tokens_2)

    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio


if __name__ == "__main__":
    # print ratio as a pandas dataframe
    # row = method, baseline is all 1
    # column = dataset
    # model = "gpt-5-mini"
    model = "gpt-5.4"
    methods = ['react_agent_clean']
    results = []
    for dataset in DATASET_LIST:
        for method in methods:
            # avg_ratio = avg_input_token_usage_ratio(dataset, method, 'baseline', model)
            # avg_ratio = avg_output_token_usage_ratio(dataset, method, 'baseline', model)
            avg_ratio = avg_input_cost_ratio(dataset, method, 'baseline', model)
            results.append({
                'dataset': dataset,
                'method': method,
                'avg_ratio': avg_ratio
            })
    df = pd.DataFrame(results)

    df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    df = df.sort_values('method')
    # print(df.to_csv(index=False))
    # print as csv
    print(df.pivot(index='method', columns='dataset', values='avg_ratio').round(4).to_csv())


