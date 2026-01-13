import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List
import json
import re
import config
import argparse
import eval.eval_civic
import eval.eval_contract
import eval.eval_qasper
import eval.eval_finance
import logging
import logging_config
logging.disable(level=logging.DEBUG)
import traceback



def get_eval_result(context_config, dataset, metric):
    m_metric_index = { # for civic2
        "recall": 0,
        "precision": 1,
        "f1": 2
    }
    if dataset == "civic1":
        return eval.eval_civic.civic_q1_eval_answer(context_config)
    elif dataset == "civic2":
        return eval.eval_civic.civic_q2_eval_answer(context_config)[m_metric_index[metric]]
    elif dataset == "civic":
        eval_civic_1 = eval.eval_civic.civic_q1_eval_answer(context_config) # 380 queries
        eval_civic_2 = eval.eval_civic.civic_q2_eval_answer(context_config)[m_metric_index[metric]] # 38 queries
        return round((eval_civic_1 * 380 + eval_civic_2 * 38) / 418, 3)

    elif dataset == "contract":
        return eval.eval_contract.contract_eval_answer(context_config)
    elif dataset == "qasper":
        if metric == "token":
            return eval.eval_qasper.qasper_eval_answer_f1(context_config)
        elif metric == "llmjudge":
            return eval.eval_qasper.qasper_eval_answer_llm(context_config)
        else:
            assert False, f"Metric {metric} not supported for Qasper."
        return eval.eval_qasper.qasper_eval_answer_llm(context_config)
    elif dataset == "finance":
        return eval.eval_finance.finance_eval_answer_llm(context_config)
    else:
        raise ValueError(f"Dataset {dataset} not supported yet.")
    


def ablation(mode: str):
    """ Data used in ablation studies"""
    orig_model_config = ("sht", None, "sbert", True, True, True, 0.2)
    
    tab_str = ""

    if mode == "sht":
        use_relative = True
        context_config_list = (
            [("sht", "intrinsic", "sbert", True, True, True, 0.2)] +
            [("sht", "deep", "sbert", True, True, True, 0.2)] + 
            [("sht", "wide", "sbert", True, True, True, 0.2)] + 
            [("sht", "grobid", "sbert", True, True, True, 0.2)] + 
            [("sht", "llm_txt", "sbert", True, True, True, 0.2)] + 
            [("sht", "llm_vision", "sbert", True, True, True, 0.2)] +
            [("sht", None, "sbert", True, True, True, 0.2)]
        )
    elif mode == "hi":
        use_relative = True
        context_config_list = [("sht", None, "sbert", False, True, True, 0.2), ("sht", None, "sbert", True, False, True, 0.2), ("sht", None, "sbert", False, False, True, 0.2)]
    elif mode == "ci":
        use_relative = True
        context_config_list = [("sht", None, "sbert", True, True, False, 0.2)]
    elif mode == "embedding":
        use_relative = False
        context_config_list = (
            [("vanilla", None, nem, None, None, None, 0.2) for nem in config.NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
            [("raptor", None, nem, None, None, None, 0.2) for nem in config.NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
            [("hipporag", None, nem, None, None, None, 0.2) for nem in config.NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
            [("sht", None, nem, True, True, True, 0.2) for nem in config.NODE_EMBEDDING_MODEL_LIST if nem != "sbert"]
        )
    else:
        raise ValueError(f"Ablation study {mode} not supported.")

    
    for context_config in context_config_list:
        line_str = str(context_config) + " & "
        data_list = []
        for dataset in [
            "civic", 
            "contract", 
            "qasper", 
            "finance"
        ]:
            logging.info(f"Evaluating context_config={context_config}...")
            metric = "llmjudge" if dataset in ["qasper", "finance"] else ("f1" if "civic" in dataset else None)
            result = round(get_eval_result(context_config, dataset, metric), 2)
            if use_relative == True:
                orig_result = round(get_eval_result(orig_model_config, dataset, metric), 2)
                relative_increase = round((100 * (result - orig_result) / orig_result), 2)
            else:
                relative_increase = round(result, 2)
            result = relative_increase
            data_list.append(result)
            line_str += str(result) + "\% & "
        tab_str += line_str.strip()[:-1] + f" & {round(sum(data_list) / len(data_list), 2)} \% " + "\\\\\n"

    print(tab_str)

def max_improvement():
    """ Find the configuration with the maximum improvement over baselines (in Abstract and Intro)"""
    max_improvement = None
    max_improvement_configs = None
    for baseline in [
        "vanilla", 
        "raptor", 
        "hipporag", 
        "graphrag"
    ]:
        for embedding_mode in [
            # 'bm25', 
            # 'dpr', 
            # 'te3small', 
            'sbert'
        ]:
            for dataset in [
                "civic", 
                "contract", 
                "qasper", 
                "finance"
            ]:
                metric = "llmjudge" if dataset in ["qasper", "finance"] else ("f1" if "civic" in dataset else None)
                context_len_ratio_list = config.CONTEXT_LEN_RATIO_LIST
                if embedding_mode != "sbert":
                    context_len_ratio_list = [0.2]
                for context_len_ratio in context_len_ratio_list:
                    opt_config = ("sht", None, embedding_mode, True, True, True, context_len_ratio)
                    base_config = (baseline, None, embedding_mode, None, None, None, context_len_ratio)
                    opt_result = get_eval_result(opt_config, dataset, metric)
                    try:
                        base_result = get_eval_result(base_config, dataset, metric)
                    except Exception as e:
                        continue
                    improvement = opt_result - base_result
                    if max_improvement == None or (max_improvement > improvement):
                        max_improvement = improvement
                        max_improvement_configs = [opt_config, base_config, dataset]
                    
                    
    print(max_improvement)
    print(max_improvement_configs)
