import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATASET_LIST, DATA_ROOT_FOLDER
from structured_rag.utils import get_nondummy_ancestors
from agents.utils import get_toc_level

MAX_OUTPUT_TOKENS = 150

DEBUG = True

def toc_format(dataset, toc_nodes):
    m_id_level = get_toc_level(dataset, toc_nodes)

    banned_type = ['text']
    if dataset != 'contract':
        banned_type.append('list')

    m_id_to_node = {node["id"]: node for node in toc_nodes}

    # format TOC string
    # print(m_id_level)
    toc_lines = ""
    assert [node['id'] for node in toc_nodes] == sorted(node['id'] for node in toc_nodes)

    for node in toc_nodes:
        if node['id'] not in m_id_level:
            continue

        level_str = ".".join(str(num).strip() for num in m_id_level[node['id']])
        if dataset != "contract":
            heading = node['heading'].strip()
        else:
            raw_heading = node['heading'].strip()
            if len(raw_heading) <= 4:
                next_id = None
                cur_id = node['id'] + 1
                while cur_id in m_id_to_node:
                    if m_id_to_node[cur_id]['is_dummy'] == False:
                        if m_id_to_node[cur_id]['type'] in banned_type: # banned_type = ['text'] for contract dataset
                            next_id = cur_id
                        break   
                    cur_id += 1
                if next_id == None:
                    heading = raw_heading
                else:
                    next_node_txt = " ".join(m_id_to_node[next_id]['texts'])
                    suffix = ""
                    if len(next_node_txt) > 80:
                        suffix = "..."
                    heading = raw_heading + " " + next_node_txt[:80] + suffix
            else:
                heading = raw_heading

        toc_lines += f"{level_str} | {heading.strip()}\n"

    return toc_lines.strip()


def generate_toc_in_context(dataset, file_name, sht_type):
    true_sht_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type, # "intrinsic",
        "sbert.gpt-4o-mini.c100.s100", 
        "sht", 
        file_name + ".json"
    )
    toc_nodes = json.load(open(true_sht_path, "r"))["nodes"]
    toc_str = toc_format(dataset, toc_nodes)

    dst_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        "toc_numbered",
    )
    os.makedirs(dst_path, exist_ok=True)
    with open(os.path.join(dst_path, file_name + ".txt"), "w") as f:
        f.write(toc_str)


if __name__ == "__main__":
    for sht_type in [
        'deep', 
        # 'wide', 
        # 'grobid', 
        # ''
        ]:
        for dataset in DATASET_LIST:
            print(f"{dataset}")
            
            pdf_folder = os.path.join(DATA_ROOT_FOLDER, dataset, "pdf")
            for file_name in sorted(os.listdir(pdf_folder)):
                file_name = file_name.replace(".pdf", "")
                print(f"\t{file_name}")
                generate_toc_in_context(dataset, file_name, sht_type)
        