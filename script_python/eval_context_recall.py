import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging_config
import numpy as np
from structured_rag import split_text_into_sentences
import eval.utils
from collections import Counter
import tiktoken
import config

def _clean_spaces(s):
    return eval.utils.white_space_fix(s).strip()

def _clean_txt(txt):
    return (" ".join([eval.utils.white_space_fix(s).strip() for s in split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], txt) if len(s.strip()) > 0]))

def _sim(sht_chunk: str, true_prov: str):
    def _norm_str(s: str):
        return eval.utils.remove_articles(eval.utils.remove_punc(eval.utils.lower(s)))
    sht_chunk_tokens = _norm_str(sht_chunk).split()
    true_prov_tokens = _norm_str(true_prov).split()
    common = Counter(sht_chunk_tokens) & Counter(true_prov_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    recall = 1.0 * num_same / len(true_prov_tokens)
    return recall

def _get_true_prov_list(query_info, dataset):
    if dataset != 'qasper':
        # context = list[str]
        assert isinstance(query_info['context'], list)
        assert all([isinstance(ctx, str) for ctx in query_info['context']])
        prov_list = query_info['context']
    else:
        # context = list[list[str]]
        assert isinstance(query_info['context'], list)
        assert all([isinstance(cl, list) for cl in query_info['context']])
        assert all([isinstance(ctx, str) for cl in query_info['context'] for ctx in cl])
        prov_list = min(query_info['context'], key=lambda x: len(x))
    
    true_prov_list = [_clean_txt(prov) for prov in prov_list]
    result = [t.strip() for t in true_prov_list if len(t.strip()) > 0]
    if len(result) == 0:
        return [_clean_txt(query_info['query'])]
    return result

def _recall(context: str, true_prov_list):
    ctx_sentences = split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], context)
    true_prov_sentences = []
    for prov in true_prov_list:
        true_prov_sentences.extend(split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], prov))
    
    ctx_sentences = [_clean_spaces(s) for s in ctx_sentences]
    true_prov_sentences = [_clean_spaces(s) for s in true_prov_sentences]
    
    ctx_sentences = [s for s in ctx_sentences if len(s) > 0]
    true_prov_sentences = [s for s in true_prov_sentences if len(s) > 0]

    true_prov_recall = []
    for prov in true_prov_sentences:
        ctx_recall_list = [
            _sim(ctx_sent, prov) for ctx_sent in ctx_sentences
        ]
        if len(ctx_recall_list) == 0:
            true_prov_recall.append(0.0)
        else:
            true_prov_recall.append(max(ctx_recall_list))
    
    return np.mean(true_prov_recall) if len(true_prov_recall) > 0 else 0.0


def recall(dataset, sht_type):
    query_path = os.path.join(
        "/home/ruiying/SHTRAG/data",
        dataset,
        "queries.json"
    )
    sht_config = ("sht", sht_type if sht_type != 'shed' else None, "sbert", True, True, True, 0.2)
    context_path = config.get_config_jsonl_path(
        dataset,
        sht_config
    )

    with open(query_path, 'r') as file:
        query_list = json.load(file)
    
    with open(context_path, 'r') as file:
        context_list = [json.loads(line) for line in file.readlines()]

    assert len(query_list) == len(context_list), (len(query_list), len(context_list))

    recall_list = []
    for query_info, context_info in zip(query_list, context_list):
        print(f"Processing query ID: {query_info['id']}")
        assert query_info['id'] == context_info['id'], (query_info['id'], context_info['id'])
        true_prov_list = _get_true_prov_list(query_info, dataset)
        context = context_info['context']
        recall_value = _recall(context, true_prov_list)
        recall_list.append(recall_value)
        print(f"  Recall: {recall_value:.4f}")

    return np.mean(recall_list) if len(recall_list) > 0 else 0.0

if __name__ == "__main__":
    dataset = "finance"
    m_sht_recall = dict()
    sht_type_list = [
        "intrinsic",
        "grobid",
        "wide",
        "deep",
        "llm_txt",
        "llm_vision",
        'shed'
    ]
    for sht_type in sht_type_list:
        recall_value = recall(dataset, sht_type)
        m_sht_recall[sht_type] = recall_value
        print(f"SHT Type: {sht_type if sht_type is not None else 'default'}, Recall: {recall_value:.4f}")
    
    print("Summary of SHT Recall:")
    for sht_type, recall_value in m_sht_recall.items():
        print(f"SHT Type: {sht_type if sht_type is not None else 'default'}, Recall: {recall_value:.4f}")