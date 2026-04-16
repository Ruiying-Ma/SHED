import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from structured_rag.utils import get_nondummy_ancestors, get_grep_text
from agents.utils import get_toc_level
import traceback

MAX_OUTPUT_TOKENS = 150

DEBUG = True

def toc_grep_text(dataset, file_name, sht_type):
    true_sht_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type, #"intrinsic",
        "sbert.gpt-4o-mini.c100.s100", 
        "sht" if dataset != 'office' else "sht_skeleton",
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
        text_span = get_grep_text(toc_nodes, node_id, dataset)
        m_level_textspan[level_str] = text_span
    
    dst_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_greptext",
        file_name + ".json"
    )
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as file:
        json.dump(m_level_textspan, file, indent=2)

if __name__ == "__main__":
    for sht_type in [
        # 'grobid', 
        # '',
        # "llm_txt_sht",
        'intrinsic'
    ]:
        for dataset in DATASET_LIST[:-1]: # finance
        # for dataset in ['office']:
            print(f"{dataset}")
            
            pdf_folder = os.path.join(DATA_ROOT_FOLDER, dataset, "pdf")
            for file_name in sorted(os.listdir(pdf_folder)):
                file_name = file_name.replace(".pdf", "")
                print(f"\t{file_name}")
                try:
                    toc_grep_text(dataset, file_name, sht_type)
                except Exception as e:
                    logging.warning(f"{sht_type} {dataset} {file_name} failed; Traceback:\n{traceback.format_exc()}")