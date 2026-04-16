import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time
from typing import List, Dict

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_toc_level
from agents.listing import strip_leading_numbering
from structured_rag.utils import get_nondummy_ancestors, get_successor_head_id
import traceback

MAX_OUTPUT_TOKENS = 150

DEBUG = True

def get_textspan_clean(nodes: List[Dict], node_id: int, dataset):
    assert [n['id'] for n in nodes] == list(range(len(nodes))), f"{[n['id'] for n in nodes]}\n{len(nodes)}"
    
    if dataset != 'contract':
        head_types = ['head']
    else:
        head_types = ['head', 'list']

    assert nodes[node_id]['type'] in head_types, f"{nodes[node_id]['type']} not in {head_types}"

    next_head_id = get_successor_head_id(nodes, node_id, head_types)
    textspan = ""

    for ni in range(node_id, next_head_id):
        if nodes[ni]['is_dummy'] == False:
            if nodes[ni]['type'] == 'text':
                textspan += " ".join(nodes[ni]['texts']).strip() + "\n"
            else:
                if ni == node_id:
                    if dataset in ['contract_rand_v0_1', 'finance_rand_v1', 'qasper_rand_v1']:
                        textspan += strip_leading_numbering(nodes[ni]['heading'].strip())
                    elif dataset == 'civic_rand_v1':
                        textspan += nodes[ni]['heading'].strip()
                else:
                    textspan += nodes[ni]['heading'].strip()
                if nodes[ni]['type'] == 'head':
                    textspan += "\n"

    return textspan.strip()

def toc_textspan(dataset, file_name, sht_type):
    true_sht_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type, #"intrinsic",
        "sbert.gpt-4o-mini.c100.s100", 
        "sht", 
        file_name + ".json"
    )
    if not os.path.exists(true_sht_path):
        true_sht_path = true_sht_path.replace("/sht/", "/sht_skeleton/")
    toc_nodes = json.load(open(true_sht_path, "r"))["nodes"]

    m_id_level = get_toc_level(dataset, toc_nodes)
    assert [node['id'] for node in toc_nodes] == list(range(len(toc_nodes)))

    banned_type = ['text']
    if dataset != 'contract':
        banned_type.append('list')

    m_level_textspan = dict()
    for node_id, level in m_id_level.items():
        if node_id == -1:
            continue
        level_str = ".".join(str(num).strip() for num in level)
        assert str(level_str) not in m_level_textspan
        text_span = get_textspan_clean(toc_nodes, node_id, dataset)
        m_level_textspan[level_str] = text_span
    
    dst_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_textspan_clean",
        file_name + ".json"
    )
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as file:
        json.dump(m_level_textspan, file, indent=2)

if __name__ == "__main__":
    for sht_type in [
        # 'deep', 
        # 'wide', 
        # 'grobid', 
        # '',
        "llm_txt_sht",
        "llm_vision_sht"
    ]:
        for dataset in DATASET_LIST[2:]: # finance
        # for dataset in ['office']:
            print(f"{dataset}")

            node_clustering_folder = os.path.join(
                DATA_ROOT_FOLDER,
                dataset,
                sht_type,
                "node_clustering"
            )

            if dataset == 'finance_rand_v1':
                start_id = 74
                end_id = 100
            elif dataset == 'qasper_rand_v1':
                start_id = 290
                end_id = 500
            
            # for file_name in sorted(os.listdir(node_clustering_folder)):
            for file_id in range(start_id, end_id):
                file_name = str(file_id)
                # file_name = file_name.replace(".json", "")
                print(f"\t{file_name}")
                try:
                    toc_textspan(dataset, file_name, sht_type)
                except Exception as e:
                    logging.warning(f"{sht_type} {dataset} {file_name} failed; Traceback:\n{traceback.format_exc()}")