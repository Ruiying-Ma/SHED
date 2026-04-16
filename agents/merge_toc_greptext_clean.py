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
from structured_rag.utils import get_grep_successor_head_id
from agents.utils import get_toc_level, get_doc_txt
from agents.listing import strip_leading_numbering
import traceback
import unicodedata
import string
import calendar
import hashlib

MAX_OUTPUT_TOKENS = 150

DEBUG = True

def hash8_digits(s: str) -> str:
    h = hashlib.sha256(s.encode()).hexdigest()
    num = int(h, 16)          # convert hex → integer
    return str(num % 10**8).zfill(8).lower()

def normalize_string(text: str):
    return unicodedata.normalize('NFKC', text)

def white_space_fix(text):
    return " ".join(text.split())

def replace_punc(text, rep = " "):
    exclude = set(string.punctuation)
    # replace punctuation with space
    return "".join(ch if ch not in exclude else rep for ch in text)

def lower(text):
    return text.lower()

def format_doc_name(doc_name, dataset):
    if dataset == "civic":
        # doc_name is in the format of "MMDDYYYY"
        month = int(doc_name[:2])
        day = int(doc_name[2:4])
        year = int(doc_name[4:8])
        month_name = calendar.month_name[month]
        return f"Meeting: {month_name} {day}, {year}"
    elif dataset == 'contract':
        return replace_punc(white_space_fix(normalize_string(doc_name))).capitalize()
    elif dataset == 'finance':
        # get a fixed hash of the doc_name with 8 digits
        name_hash = hash8_digits(doc_name)
        return f"Financial document {name_hash.lower()}"
    elif dataset == 'qasper':
        with open(f"/home/ruiying/SHTRAG/data/qasper/grobid/grobid/{doc_name}.json", 'r') as file:
            grobid_info = json.load(file)
        title = grobid_info['title']
        return title
    else:
        raise NotImplementedError(f"Doc name formatting for dataset {dataset} not implemented yet")


def format_doc_level_str(doc_level):
    return ".".join(str(num).strip() for num in doc_level)


def get_grep_text_clean(nodes: List[Dict], node_id: int, dataset):
    assert [n['id'] for n in nodes] == list(range(len(nodes))), f"{[n['id'] for n in nodes]}\n{len(nodes)}"
    
    if dataset != 'contract':
        head_types = ['head']
    else:
        head_types = ['head', 'list']

    assert nodes[node_id]['type'] in head_types, f"{nodes[node_id]['type']} not in {head_types}"

    next_head_id = get_grep_successor_head_id(nodes, node_id, head_types)
    textspan = ""

    if dataset in ['contract', 'finance', 'qasper']:
        heading = strip_leading_numbering(nodes[node_id]['heading'].strip())
    elif dataset == 'civic':
        heading = nodes[node_id]['heading'].strip()

    for ni in range(node_id + 1, next_head_id):
        if nodes[ni]['is_dummy'] == False:
            if nodes[ni]['type'] == 'text':
                textspan += " ".join(nodes[ni]['texts']).strip() + "\n"
            else:
                assert nodes[ni]['type'] not in head_types
                textspan += nodes[ni]['heading'].strip()
                if nodes[ni]['type'] == 'head':
                    textspan += "\n"

    return {'heading': heading, 'text': textspan.strip()}

def orig_toc_grep_text_per_doc(dataset, toc_nodes, sht_type):
    m_id_level = get_toc_level(dataset, toc_nodes)
    assert [node['id'] for node in toc_nodes] == list(range(len(toc_nodes)))

    banned_type = ['text']
    if dataset != 'contract':
        banned_type.append('list')

    # m_level_textspan = dict()
    # for node_id, level in m_id_level.items():
    #     if node_id == -1:
    #         continue
    #     level_str = ".".join(str(num).strip() for num in level)
    #     assert str(level_str) not in m_level_textspan
    #     text_span = get_grep_text_clean(toc_nodes, node_id, dataset)
    #     m_level_textspan[level_str] = text_span

    m_level_textspan = dict()
    for node_id, level in m_id_level.items():
        if node_id == -1:
            continue
        text_span = get_grep_text_clean(toc_nodes, node_id, dataset)
        assert tuple(level) not in m_level_textspan, f"{tuple(level)} already in m_level_textspan for node_id {node_id} with text_span:\n{text_span}\n"
        m_level_textspan[tuple(level)] = text_span
    
    return m_level_textspan



def merge_toc_textspans(sht_type, m_doc_map_level_textspan, orig_dataset):
    if orig_dataset == 'civic':
        total_header = "Agenda Reports for Civic Projects"
    elif orig_dataset == 'contract':
        total_header = "Non-Disclosure Agreements"
    elif orig_dataset == 'finance':
        total_header = "Financial Documents"
    elif orig_dataset == 'qasper':
        total_header = "Papers"
    else:
        raise NotImplementedError(f"Total header for dataset {orig_dataset} not implemented yet")
    

    if sht_type == 'wide':
        
        map_level_textspan = {
            "1": {"heading": total_header, "text": ""},
        }

        for doc, map_level_ts in m_doc_map_level_textspan.items():
            doc_level = format_doc_level_str(tuple([1, len(map_level_textspan)]))
            map_level_textspan[doc_level] = {
                "heading": format_doc_name(doc, orig_dataset),
                "text": "",
            }
            if len(map_level_ts) == 0:
                map_level_textspan[doc_level]['text'] += get_doc_txt(orig_dataset, doc)
            # sorted ts by level
            sorted_level_ts = sorted(map_level_ts.items(), key=lambda x: x[0])
            for level, ts in sorted_level_ts:
                new_level = format_doc_level_str(tuple([1, len(map_level_textspan)]))
                map_level_textspan[new_level] = ts
        return map_level_textspan

    elif sht_type in ['', 'intrinsic', 'grobid', 'llm_txt', 'llm_vision']:
        map_level_textspan = {
            "1": {"heading": total_header, "text": ""},
        }
        doc_cnt = 1
        for doc, map_level_ts in m_doc_map_level_textspan.items():
            doc_level = tuple([1, doc_cnt])
            map_level_textspan[format_doc_level_str(doc_level)] = {
                "heading": format_doc_name(doc, orig_dataset),
                "text": ""
            }
            if len(map_level_ts) == 0:
                map_level_textspan[format_doc_level_str(doc_level)]['text'] += get_doc_txt(orig_dataset, doc)
            # sorted ts by level
            sorted_level_ts = sorted(map_level_ts.items(), key=lambda x: x[0])
            for level, ts in sorted_level_ts:
                new_level = tuple(list(doc_level) + list(level))
                map_level_textspan[format_doc_level_str(new_level)] = ts
            doc_cnt += 1
        return map_level_textspan
            

    else:
        raise NotImplementedError(f"Merge TOC lines for sht type {sht_type} not implemented yet")
    
def merge_toc_textspan_per_query(orig_dataset, new_file_name, sht_type, new_file_names, merged_dataset, need_store=True):
    m_doc_map_textspans = dict()
    for fid, file_name in enumerate(new_file_names):
        true_sht_path = os.path.join(
            DATA_ROOT_FOLDER,
            orig_dataset,
            sht_type, # "intrinsic",
            "sbert.gpt-4o-mini.c100.s100", 
            "sht", 
            file_name + ".json"
        )
        if not os.path.exists(true_sht_path):
            true_sht_path = true_sht_path.replace("/sht/", "/sht_skeleton/")
            # assert os.path.exists(true_sht_path), f"No SHT or SHT skeleton found for {dataset} {file_name} at {true_sht_path}"
        
        if not os.path.exists(true_sht_path):
            print(f"No SHT or SHT skeleton found for {orig_dataset} {file_name} at {true_sht_path}")
            return
        
        toc_nodes = json.load(open(true_sht_path, "r"))["nodes"]
        map_toc_ts = orig_toc_grep_text_per_doc(orig_dataset, toc_nodes, sht_type) # {level: textspan}
        m_doc_map_textspans[file_name] = map_toc_ts

    merged_textspans = merge_toc_textspans(sht_type, m_doc_map_textspans, orig_dataset)

    if need_store:
        dst_path = os.path.join(
            DATA_ROOT_FOLDER,
            merged_dataset,
            sht_type,
            "toc_greptext_clean",
        )
        os.makedirs(dst_path, exist_ok=True)
        with open(os.path.join(dst_path, new_file_name + ".json"), "w") as f:
            json.dump(merged_textspans, f, indent=2)


if __name__ == "__main__":
    orig_dataset = "qasper"
    merged_dataset = "qasper_rand_v1"
    with open(f"/home/ruiying/SHTRAG/data/{merged_dataset}/queries.json", 'r') as file:
        queries = json.load(file)
    
    for sht_type in [
        # 'deep', 
        'wide', 
        'grobid', 
        '',
        # "llm_txt_sht",
        "intrinsic"
    ]:
        existing_file_names = set()
        for qinfo in queries[290:500]:
            file_name = qinfo['file_name']
            assert file_name not in existing_file_names, f"Duplicate file name {file_name} in queries.json"
            existing_file_names.add(file_name)
            new_file_names = qinfo['new_file_names']
            print(f"query {qinfo['id']}")
            merge_toc_textspan_per_query(
                orig_dataset=orig_dataset,
                new_file_name=file_name,
                sht_type=sht_type,
                new_file_names=new_file_names,
                merged_dataset=merged_dataset
            )
       
        