import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time
import calendar
import random

from config import DATASET_LIST, DATA_ROOT_FOLDER
from structured_rag.utils import get_nondummy_ancestors
from agents.utils import get_toc_level
from agents.listing import strip_leading_numbering
import unicodedata
import string
import hashlib

logging.getLogger().setLevel(logging.WARNING)



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

def format_level_str(level):
    return ".".join(str(num).strip() for num in level)

def format_toc_lines(map_level_heading):
    toc_lines = ""
    for level, heading in map_level_heading.items():
        level_str = format_level_str(level)
        toc_lines += f"{level_str} | {heading.strip()}\n"
    return toc_lines

def remove_company_name(file_name, txt: str):
    assert txt == txt.capitalize()
    company_name_cap = file_name.split("_")[0]
    if company_name_cap == "ACTIVISIONBLIZZARD":
        company_names = ["activision blizzard, inc.", "activision blizzard"]
    elif company_name_cap == "ADOBE":
        company_names = ["adobe systems incorporated", "adobe"]
    elif company_name_cap == "AES":
        company_names = ["the aes corporation", "aes"]
    elif company_name_cap == "AMAZON":
        company_names = ["amazon.com"]
    elif company_name_cap == "AMCOR":
        company_names = ["amcor plc", "amcor"]
    elif company_name_cap == "AMD":
        company_names = ["advanced micro devices, inc."]
    elif company_name_cap == "AMERICANEXPRESS":
        company_names = ["american express company", "american express"]
    elif company_name_cap == "AMERICANWATERWORKS":
        company_names = ["american water works company, inc."]
    elif company_name_cap == "BESTBUY":
        company_names = ["best buy", "Best buy co., inc."]
    elif company_name_cap == "BLOCK":
        company_names = ["Square, inc."]
    elif company_name_cap == "COCACOLA":
        company_names = ["coca-cola"]
    elif company_name_cap == "CVSHEALTH":
        company_names = ["cvs health corporation", "cvs health"]
    elif company_name_cap == "GENERALMILLS":
        company_names = ["general mills, inc."]
    elif company_name_cap == "JOHNSON":
        company_names = ["johnson & johnson"]
    elif company_name_cap == "JPMORGAN":
        company_names = ["jpmorgan chase & co.", "jpmorgan chase"]
    elif company_name_cap == "KRAFTHEINZ":
        company_names = ["kraft heinz"]
    elif company_name_cap == "LOCKHEEDMARTIN":
        company_names = ["lockheed martin corporation"]
    elif company_name_cap == "MGMRESORTS":
        company_names = ["mgm resorts international", "mgm r esorts i nternational", "mgm"]
    elif company_name_cap == "NETFLIX":
        company_names = ["netflix, inc."]
    elif company_name_cap == "NIKE":
        company_names = ["nike, inc.", "nike"]
    elif company_name_cap == "PAYPAL":
        company_names = ["paypal holdings, inc.", "paypal"]
    elif company_name_cap == "PEPSICO":
        company_names = ["pepsico, inc.", "pepsico"]
    elif company_name_cap == "PFIZER":
        company_names = ["pfizer inc.", "pfizer"]
    elif company_name_cap == "ULTABEAUTY":
        company_names = ["ulta beauty, inc."]
    elif company_name_cap == "VERIZON":
        company_names = ["verizon"]
    elif company_name_cap == "WALMART":
        company_names = ["walmart inc.", "walmart"]
    else:
        company_names = [company_name_cap.lower()]

    sorted_company_names = sorted(company_names, key=lambda x: len(x), reverse=True)
    candid_txt = txt.lower()
    for company_name in sorted_company_names:
        candid_txt = candid_txt.replace(company_name, "")
    
    return candid_txt.capitalize()


def orig_toc_format_per_doc(dataset, toc_nodes, file_name):
    m_id_level = get_toc_level(dataset, toc_nodes)

    banned_type = ['text']
    if dataset != 'contract':
        banned_type.append('list')

    m_id_to_node = {node["id"]: node for node in toc_nodes}

    # format TOC string
    # print(m_id_level)
    map_toc_lines = dict() # {level: heading}
    assert [node['id'] for node in toc_nodes] == sorted(node['id'] for node in toc_nodes)

    for node in toc_nodes:
        if node['id'] not in m_id_level:
            continue

        # level_str = ".".join(str(num).strip() for num in m_id_level[node['id']])
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

        # toc_lines += f"{level_str} | {strip_leading_numbering(heading.strip())}\n"
        level_key = tuple(m_id_level[node['id']])
        assert level_key not in map_toc_lines, f"Duplicate level {level_key} in TOC for dataset {dataset}"
        if dataset == 'contract':
            map_toc_lines[level_key] = strip_leading_numbering(heading.strip())
        elif dataset == 'civic':
            map_toc_lines[level_key] = heading.strip()
        elif dataset == 'finance':
            map_toc_lines[level_key] = remove_company_name(file_name, strip_leading_numbering(heading.strip()))
        elif dataset == 'qasper':
            map_toc_lines[level_key] = strip_leading_numbering(heading.strip())
        else:
            raise NotImplementedError(f"Unknown dataset {dataset}")

    
    return map_toc_lines

def merge_toc_lines(sht_type, m_doc_map_toc_lines, orig_dataset):
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
        
        map_toc_lines = {
            tuple([1]): total_header,
        }
        for doc, map_level_heading in m_doc_map_toc_lines.items():
            doc_level = tuple([1, len(map_toc_lines)])
            map_toc_lines[doc_level] = format_doc_name(doc, orig_dataset)
            # sorted headings by level
            sorted_level_heading = sorted(map_level_heading.items(), key=lambda x: x[0])
            for level, heading in sorted_level_heading:
                new_level = tuple([1, len(map_toc_lines)])
                map_toc_lines[new_level] = heading
        return format_toc_lines(map_toc_lines)
    
    elif sht_type == 'deep':
        map_toc_lines = {
            tuple([1]): total_header,
        }
        for doc, map_level_heading in m_doc_map_toc_lines.items():
            doc_level = tuple([1 for _ in range(len(map_toc_lines) + 1)])
            map_toc_lines[doc_level] = format_doc_name(doc, orig_dataset)
            # sorted headings by level
            sorted_level_heading = sorted(map_level_heading.items(), key=lambda x: x[0])
            for level, heading in sorted_level_heading:
                new_level = tuple(list(doc_level) + list(level))
                map_toc_lines[new_level] = heading
        return format_toc_lines(map_toc_lines)

    elif sht_type in ['', 'intrinsic', 'grobid', 'llm_txt', 'llm_vision']:
        map_toc_lines = {
            tuple([1]): total_header,
        }
        doc_cnt = 1
        for doc, map_level_heading in m_doc_map_toc_lines.items():
            doc_level = tuple([1, doc_cnt])
            map_toc_lines[doc_level] = format_doc_name(doc, orig_dataset)
            # sorted headings by level
            sorted_level_heading = sorted(map_level_heading.items(), key=lambda x: x[0])
            for level, heading in sorted_level_heading:
                new_level = tuple(list(doc_level) + list(level))
                map_toc_lines[new_level] = heading
            doc_cnt += 1
        return format_toc_lines(map_toc_lines)
            

    else:
        raise NotImplementedError(f"Merge TOC lines for sht type {sht_type} not implemented yet")
                
            

def merge_toc_in_context_per_query(orig_dataset, new_file_name, sht_type, new_file_names, merged_dataset, need_store=True):
    m_doc_map_toc_lines = dict()
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
        map_toc_lines = orig_toc_format_per_doc(orig_dataset, toc_nodes, file_name) # {level: heading}
        m_doc_map_toc_lines[file_name] = map_toc_lines

    merged_toc_str = merge_toc_lines(sht_type, m_doc_map_toc_lines, orig_dataset)

    if need_store:
        dst_path = os.path.join(
            DATA_ROOT_FOLDER,
            merged_dataset,
            sht_type,
            "toc_numbered_clean",
        )
        os.makedirs(dst_path, exist_ok=True)
        with open(os.path.join(dst_path, new_file_name + ".txt"), "w") as f:
            f.write(merged_toc_str)


if __name__ == "__main__":
    orig_dataset = "qasper"
    merged_dataset = "qasper_rand_v1"
    with open(f"/home/ruiying/SHTRAG/data/{merged_dataset}/queries.json", 'r') as file:
        queries = json.load(file)
    
    for sht_type in [
        'deep', 
        'wide', 
        'grobid', 
        '',
        # "llm_txt_sht",
        # "llm_vision_sht",
        "intrinsic"
    ]:
        existing_file_names = set()
        for qinfo in queries[290:500]:
            file_name = qinfo['file_name']
            assert file_name not in existing_file_names, f"Duplicate file name {file_name} in queries.json"
            existing_file_names.add(file_name)
            new_file_names = qinfo['new_file_names']
            print(f"query {qinfo['id']}")
            merge_toc_in_context_per_query(
                orig_dataset=orig_dataset,
                new_file_name=file_name,
                sht_type=sht_type,
                new_file_names=new_file_names,
                merged_dataset=merged_dataset
            )
       
        