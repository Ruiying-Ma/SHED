from pathlib import Path
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import TypedDict
import time
import json
from datetime import datetime
import re

from config import DATA_ROOT_FOLDER, DATASET_LIST, get_cost_usd
from structured_rag.utils import get_nondummy_ancestors

class LLMResponse(TypedDict):
    is_success: bool
    message: str
    latency: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int

load_dotenv(Path(__file__).parent.parent / ".env")

CLIENT = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
)

RESULTS_DIR = Path(__file__).parent / "results"

def get_result_path(dataset, model, method, sht_type):
    M_SHT_NAME = {
        'intrinsic': 'true_sht',
        'deep': 'deep_sht',
        'wide': 'wide_sht',
        'grobid': 'grobid_sht',
        '': 'shed_sht',
        'llm_txt_sht': "llm_txt_sht",
        'llm_vision_sht': "llm_vision_sht",
    }
    assert sht_type in M_SHT_NAME, f"{sht_type} is not a valid sht_type"
    if method == "baseline":
        result_path = RESULTS_DIR / "core" / model / method / f"{dataset}.jsonl"
    else:
        result_path = RESULTS_DIR / "core" / model / M_SHT_NAME[sht_type] / method / f"{dataset}.jsonl"
    os.makedirs(result_path.parent, exist_ok=True)
    return result_path

def llm(messages, model, reasoning_effort=None):
    if model == "gpt-5-mini":
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
        )
    elif model == "gpt-5.4":
        if reasoning_effort != None:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
        else:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
            )
    else:
        raise NotImplementedError(f"model {model} is not supported")
    return response

def get_llm_response(messages, model, reasoning_effort=None) -> LLMResponse:
    start_time = time.time()
    try:
        response = llm(messages, model, reasoning_effort)
    except Exception as e:
        return LLMResponse(
            is_success=False,
            message=f"{type(e).__name__}: {str(e)}",
            latency=0.0,
            input_tokens=0,
            cached_tokens=0,
            output_tokens=0,
        )
    
    end_time = time.time()

    answer = response.choices[0].message.content.strip()
    input_tokens = response.usage.prompt_tokens
    cached_tokens = response.usage.prompt_tokens_details.cached_tokens
    input_tokens = input_tokens - cached_tokens
    output_tokens = response.usage.completion_tokens
    return LLMResponse(
        is_success=True,
        message=answer,
        latency=end_time - start_time,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
    )

def get_system_message(dataset):
    # if dataset == "civic":
    #     msg = "You are an assistant for analyzing government agenda reports on civic projects. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "HINTS:\n" \
    #     "- The status of a project must be one of the following: 'not started', design', 'construction', 'completed'.\n" \
    #     "- The type of a project must be one of the following: 'capital', 'disaster'.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- If the query cannot be answered based on the provided document, respond with 'none'.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "contract":
    #     msg = "You are a helpful assistant for analyzing Non-Disclosure Agreement (NDA) contracts. Your task is to determine whether a given hypothesis is supported by the contract.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- If the hypothesis is supported by the contract, respond with: `Entailment`\n" \
    #     "- If the hypothesis is contradicted by the contract, respond with: `Contradiction`\n" \
    #     "- If the hypothesis is not mentioned in the contract, respond with: `NotMentioned`\n" \
    #     "- Use only the information provided in the contract.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "finance":
    #     msg = "You are a helpful assistant for analyzing financial documents. Your task is to answer a query using only the information provided in the documents.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- Your answer must contain no more than 3 sentences.\n" \
    #     "- Use only the information in the documents.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "finance_rand":
    #     msg = "You are a helpful assistant for analyzing financial documents. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- Your answer must contain no more than 3 sentences.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    if dataset == "finance_rand_v1":
        msg = "You are a helpful assistant for analyzing financial documents. Your task is to answer a query using only the information provided in the document.\n\n" \
        "INSTRUCTIONS:\n" \
        "- Your answer must contain no more than 3 sentences.\n" \
        "- Use only the information in the document.\n" \
        "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "qasper":
    #     msg = "You are a helpful assistant for analyzing research papers in Natural Language Processing (NLP). Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- Your answer must contain no more than 3 sentences.\n" \
    #     "- If the query cannot be answered based on the provided document, respond with 'Unanswerable'.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    elif dataset == "qasper_rand_v1":
        msg = "You are a helpful assistant for analyzing research papers in Natural Language Processing (NLP). Your task is to answer a query using only the information provided in the document.\n\n" \
        "INSTRUCTIONS:\n" \
        "- Your answer must contain no more than 3 sentences.\n" \
        "- If the query cannot be answered based on the provided document, respond with 'Unanswerable'.\n" \
        "- Use only the information in the document.\n" \
        "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "civic_new":
    #     msg = "You are an helpful assistant for analyzing government agenda reports on civic projects. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     # "HINTS:\n" \
    #     # "- The status of a project must be one of the following: 'not started', design', 'construction', 'completed'.\n" \
    #     # "- The type of a project must be one of the following: 'capital', 'disaster'.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- If the query cannot be answered based on the provided document, respond with 'none'.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == 'office':
    #     msg = "You are a helpful assistant for analyzing treasury bulletins. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- Your answer must contain no more than 3 sentences.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset in ["contract_new", 'contract_rand', 'contract_rand_v1', 'contract_rand_v2', 'contract_rand_v3']:
    #     msg = "You are a helpful assistant for analyzing Non-Disclosure Agreement (NDA) contracts. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- You response must be one of the following three options: `Entailment`, `Contradiction` or `NotMentioned`.\n" \
    #     "- Use only the information provided in the contract.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    elif dataset in ['contract_rand_v0_1']:
        msg = "You are a helpful assistant for analyzing Non-Disclosure Agreement (NDA) contracts. Your task is to answer a query using only the information provided in the document.\n\n" \
        "INSTRUCTIONS:\n" \
        "- If the query cannot be answered based on the provided document, respond with 'none'.\n" \
        "- Use only the information in the document.\n" \
        "- Return only the exact answer to the query; do not include any additional text."
    # elif dataset == "civic_rand":
    #     msg = "You are an helpful assistant for analyzing government agenda reports on civic projects. Your task is to answer a query using only the information provided in the document.\n\n" \
    #     "INSTRUCTIONS:\n" \
    #     "- If the query cannot be answered based on the provided document, respond with 'none'.\n" \
    #     "- Use only the information in the document.\n" \
    #     "- Return only the exact answer to the query; do not include any additional text."
    elif dataset == "civic_rand_v1":
        msg = "You are an helpful assistant for analyzing government agenda reports on civic projects. Your task is to answer a query using only the information provided in the document.\n\n" \
        "INSTRUCTIONS:\n" \
        "- If the query cannot be answered based on the provided document, respond with 'none'.\n" \
        "- Use only the information in the document.\n" \
        "- Return only the exact answer to the query; do not include any additional text."
    else:
        raise NotImplementedError(f"dataset {dataset} is not supported")
    
    return msg.strip()
        
def get_baseline_messages(dataset, query, document):
    system_msg = get_system_message(dataset)
    prefix = "QUERY"
    if dataset == "contract":
        prefix = "HYPOTHESIS"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{prefix}:\n{query.strip()}\n\nDOCUMENT:\n{document.strip()}"}
    ]
    return messages


            
def get_toc_level(dataset, toc_nodes):
    """
    Map each TOC node ID to its hierarchy level, represented as a list of integers. The hierarchy level is determined by the node's position in the TOC tree, with the root node having an empty list as its level. For example, if a node is the second child of the root, its level would be [2]. If it is the first child of that node, its level would be [2, 1], and so on.

    The TOC nodes only contain non-dummy head nodes (for contract, we also include list nodes)
    """
    m_id_to_node = {node["id"]: node for node in toc_nodes}
    m_parent_span = {
        -1: 0
    }
    m_id_level = {
        -1: []
    }
    banned_type = ['text']
    if dataset != 'contract':
        banned_type.append('list')

    for node in toc_nodes:
        if node['is_dummy'] == True:
            continue
        if node['type'] in banned_type:
            continue

        assert node['heading'].strip() != ""
        assert node['id'] not in m_parent_span
        assert node['id'] not in m_id_level

        nondummy_ancestors = get_nondummy_ancestors(toc_nodes, node['id'])
        if len(nondummy_ancestors) == 0:
            parent_id = -1
        else:
            # parent_id = nondummy_ancestors[-1]
            parent_id = -1
            for tmp_id in nondummy_ancestors[::-1]:
                if m_id_to_node[tmp_id]['is_dummy'] == False and m_id_to_node[tmp_id]['type'] not in banned_type:
                    parent_id = tmp_id
                    break
            if parent_id != -1:
                assert m_id_to_node[parent_id]['type'] not in banned_type
                assert m_id_to_node[parent_id]['is_dummy'] == False
            # assert node['id'] in m_id_to_node[parent_id]['nondummy_children']
            # assert m_parent_span[parent_id] == [nid for nid in m_id_to_node[parent_id]['nondummy_children'] if (m_id_to_node[nid]['is_dummy'] == False and m_id_to_node[nid]['type'] not in banned_type)].index(node['id'])
        
        assert parent_id in m_parent_span
        assert parent_id in m_id_level

        parent_level = m_id_level[parent_id]
        level = [l for l in parent_level] + [m_parent_span[parent_id] + 1]
        m_parent_span[parent_id] += 1
        m_parent_span[node['id']] = 0
        m_id_level[node['id']] = level

    return m_id_level


def get_toc_system_message(verbose: bool):
    end_first_sent = "." if verbose == False else ", which appears before the document."
    system_msg = f"""You are also given the document's table of contents (TOC){end_first_sent} 
    
TOC INSTRUCTIONS:
- Each line of TOC represents a single section in the document.
- The lines are listed in the document's reading order.
- Each line begins with a number prefix that indicates the section's hierarchy level. The prefix is separated from the section title by `|`.
- The section title itself may contain numbers. The hierarchy level is determined **only** by the number prefix, **not** by any numbers in the section title.

**Example:**

```
1 | Title
2 | 1 Section A
2.1 | 1.1 Subsection A1
2 | Section B
```

- `1 | Title` indicates that "Title" is the top-level section.
- `2 | 1 Section A` indicates that "1 Section A" is a second-level section under "Title".
- `2.1 | 1.1 Subsection A1` indicates that "1.1 Subsection A1" is a subsection under "1 Section A".
- `2 | Section B` indicates that "Section B" is another second-level section under "Title."
- Numbers inside the titles (e.g., "1 Section A" or "1.1 Subsection A1") are part of the section name and **do not affect the hierarchy**. The hierarchy is determined **only** by the number prefix before the the `|`.

You can use the TOC to understand the document's structure and help answer the query."""

    return system_msg.strip()

#  However, do not assume that all information needed to answer the query is in the TOC; it serves only as a guide to the document's organization.
    

def get_toc_in_context_messages(dataset, query, document, toc):
    system_msg = get_system_message(dataset).strip() + "\n\n" + get_toc_system_message(verbose=True).strip()
    prefix = "QUERY"
    if dataset == "contract":
        prefix = "HYPOTHESIS"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{prefix}:\n{query.strip()}\n\nTABLE OF CONTENTS (TOC):\n{toc.strip()}\n\nDOCUMENT:\n{document.strip()}"}
    ]
    return messages

def pretty_repr(messages):
    repr_str = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        repr_str += f"{'='*10} {role.upper()} {'='*10}\n{content}\n\n"
    return repr_str.strip()


def get_doc_txt(dataset, filename):
    # if dataset == 'contract_new':
    #     with open(Path(DATA_ROOT_FOLDER) / "contract_new" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    # if dataset == 'contract_rand':
    #     with open(Path(DATA_ROOT_FOLDER) / "contract_rand" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    # if dataset == 'contract_rand_v1':
    #     with open(Path(DATA_ROOT_FOLDER) / "contract_rand_v1" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    # if dataset == 'contract_rand_v2':
    #     with open(Path(DATA_ROOT_FOLDER) / "contract_rand_v2" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    # if dataset == 'contract_rand_v3':
    #     with open(Path(DATA_ROOT_FOLDER) / "contract_rand_v3" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    if dataset == 'contract_rand_v0_1':
        with open(Path(DATA_ROOT_FOLDER) / "contract_rand_v0_1" / "intrinsic" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
            full_text = json.load(file)['1'].strip()
            return full_text
    # if dataset == 'finance_rand':
    #     with open(Path(DATA_ROOT_FOLDER) / "finance_rand" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
    #         full_text = json.load(file)['1'].strip()
    #         return full_text
    if dataset == 'finance_rand_v1':
        with open(Path(DATA_ROOT_FOLDER) / "finance_rand_v1" / "intrinsic" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
            full_text = json.load(file)['1'].strip()
            return full_text
    if dataset == 'qasper_rand_v1':
        with open(Path(DATA_ROOT_FOLDER) / "qasper_rand_v1" / "intrinsic" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
            full_text = json.load(file)['1'].strip()
            return full_text
    if dataset == "finance":
        with open(Path(DATA_ROOT_FOLDER) / dataset / "doc_txt_reform_table" / (filename + ".txt"), 'r') as file:
            full_text = file.read().strip()
            return full_text
    # if dataset == 'civic_rand':
    #     with open(Path(DATA_ROOT_FOLDER) / "civic_rand" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
            # full_text = json.load(file)['1'].strip()
            # return full_text
    if dataset == 'civic_rand_v1':
        with open(Path(DATA_ROOT_FOLDER) / "civic_rand_v1" / "intrinsic" / "toc_textspan_clean" / (filename + ".json"), 'r') as file:
            full_text = json.load(file)['1'].strip()
            return full_text


    # if dataset == 'civic_new':
    #     sht_path = Path(DATA_ROOT_FOLDER) / "civic" / "sbert.gpt-4o-mini.c100.s100" / "sht" / (filename + ".json")
    # elif dataset == 'office':
    #     sht_path = Path(DATA_ROOT_FOLDER) / dataset / "sbert.gpt-4o-mini.c100.s100" / "sht_skeleton" / (filename + ".json")
    # else:
    #     sht_path = Path(DATA_ROOT_FOLDER) / dataset / "sbert.gpt-4o-mini.c100.s100" / "sht" / (filename + ".json")

    sht_path = Path(DATA_ROOT_FOLDER) / dataset / "sbert.gpt-4o-mini.c100.s100" / "sht" / (filename + ".json")

    with open(sht_path, 'r') as file:
        sht = json.load(file)

    assert "full_text" in sht
    full_text: str = sht["full_text"]

    return full_text.strip()

def get_toc_numbered(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_numbered" / (filename + ".txt")
    else:
        toc_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_numbered" / (filename + ".txt")
    return toc_path.read_text().strip()

def get_toc_numbered_clean(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_numbered_clean" / (filename + ".txt")
    else:
        toc_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_numbered_clean" / (filename + ".txt")
    return toc_path.read_text().strip()

def get_toc_textspan(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_textspan" / (filename + ".json")
    else:
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_textspan" / (filename + ".json")
    with open(toc_textspan_path, 'r') as file:
        toc_textspan = json.load(file)
    return toc_textspan


def get_toc_textspan_clean(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_textspan_clean" / (filename + ".json")
    else:
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_textspan_clean" / (filename + ".json")
    with open(toc_textspan_path, 'r') as file:
        toc_textspan = json.load(file)
    return toc_textspan

def get_toc_greptext(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_greptext" / (filename + ".json")
    else:
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_greptext" / (filename + ".json")
    with open(toc_textspan_path, 'r') as file:
        toc_textspan = json.load(file)
    return toc_textspan

def get_toc_greptext_clean(dataset, filename, sht_type):
    if dataset == "civic_new":
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / "civic" / sht_type / "toc_greptext_clean" / (filename + ".json")
    else:
        toc_textspan_path = Path(DATA_ROOT_FOLDER) / dataset / sht_type / "toc_greptext_clean" / (filename + ".json")
    with open(toc_textspan_path, 'r') as file:
        toc_textspan = json.load(file)
    return toc_textspan


def get_timestamp():
    """Return: yyyy-mm-dd__hh-mm-ss"""
    return datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

def get_simple_qasper_docs():
    """Return filenames of docs whose text already contains the hierarchical section numbers"""
    dataset = "qasper"
    simple_docs = []
    pdf_folder = Path(DATA_ROOT_FOLDER) / dataset / "pdf"
    for pdf_filename in sorted(pdf_folder.glob("*.pdf")):
        filename = pdf_filename.stem
        toc_numbered = get_toc_numbered(dataset, filename, 'intrinsic')
        numbered_title_cnt = 0
        for line in toc_numbered.splitlines():
            if line.strip() == "":
                continue
            prefix, title = line.split("|", 1)
            title = title.strip()
            # if title starts with a number, then it is already a simple doc
            if title[0].isdigit():
                numbered_title_cnt += 1
        if numbered_title_cnt >= 3:
            simple_docs.append(filename)
    
    return simple_docs

def get_unanswerable_qasper_qids():
    dataset = "qasper"
    queries = json.load(open(Path(DATA_ROOT_FOLDER) / dataset / "queries.json", 'r'))
    # unanswerable_qids = [q['id'] for q in queries if set(q['answer']) == {"Unanswerable"}]
    unanswerable_qids = [q['id'] for q in queries if "Unanswerable" in q['answer']]
    return unanswerable_qids


def get_notmentioned_contract_qids():
    dataset = "contract"
    queries = json.load(open(Path(DATA_ROOT_FOLDER) / dataset / "queries.json", 'r'))
    notmentioned_qids = [q['id'] for q in queries if q['answer'] == "NotMentioned"]
    return notmentioned_qids

def extract_numbers(text):
    """Eval script for finance dataset"""
    raw_numbers = re.findall(r'-?\d+\.?\d*', text)
    numbers = []
    for num in raw_numbers:
        if "." not in num:
            numbers.append(num)
        else:
            # remove trailing zeros and dot for float numbers
            num_clean = num.rstrip('0').rstrip('.')
            numbers.append(num_clean)
            # convert num to two decimal places, keeping 0s
            num_2dp = f"{float(num):.2f}"
            numbers.append(num_2dp)
    return numbers

def grep_search(pattern: str, toc_textspan: dict):
    sorted_section_ids = sorted(list(toc_textspan.keys()), key=lambda x: (-len(x.split(".")), x))

    matched_section_ids = []
    for section_id in sorted_section_ids:
        # check whether its children sections have matches
        has_child_match = False
        for s in matched_section_ids:
            if s.startswith(section_id + "."):
                has_child_match = True
                break
        if has_child_match == True:
            continue
        is_match = bool(re.search(pattern, toc_textspan[section_id]))
        if is_match == True:
            matched_section_ids.append(section_id)

    return sorted(matched_section_ids)

def get_hard_queries(dataset):
    query_hard_level_info_path = Path(__file__).resolve().parent.parent / "eval" / 'struct_demanding_questions' / f"{dataset}.jsonl"
    if dataset in ['civic', 'civic_new']:
        with open(Path(DATA_ROOT_FOLDER) / dataset / "queries.json", 'r') as f:
            queries = json.load(f)
            return set(q['id'] for q in queries)
    hard_qids = set()
    with open(query_hard_level_info_path, 'r') as f:
        for line in f:
            info = json.loads(line)
            if len(info['clusters']) > 0 and info['hard_level'] == 0:
                hard_qids.add(info['id'])

    return hard_qids