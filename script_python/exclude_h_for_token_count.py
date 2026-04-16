import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging_config
import numpy as np
from typing import Dict, List
import tiktoken
from structured_rag.utils import get_nondummy_ancestors, get_context_len
import config as config_utils

## only count the text nodes' tokens
## skip header nodes if they are retrieved
## but include hierarchy


def _generate_context(indexes: List[Dict], nodes: List[Dict], context_len) -> str:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    context = ""
    assert all([not nodes[i["node_id"]]["is_dummy"] for i in indexes])

    candid_nodes_for_strutcure = [] # the non-text nodes, whose headings but not texts will be added
    candid_nodes_for_text = dict()
    token_count = 0
    for index in indexes:
        node_id = index['node_id']
        chunk_id = index['chunk_id']
        node = nodes[node_id]
        assert node['type'] in ['text', 'head', 'list']

        new_text = ""
        if node['type'] != 'text':
            assert chunk_id == 0
            new_text = node['heading']
        else:
            assert node['type'] == 'text'
            assert chunk_id >=0 and chunk_id < len(node['texts'])
            new_text = node['texts'][chunk_id]
        if new_text != "":
            new_text += "\n\n"
        new_token_count = len(tokenizer.encode(new_text))

        if token_count + new_token_count <= context_len:
            ancestor_ids = get_nondummy_ancestors(nodes, node_id)
            if len(ancestor_ids) > 0:
                assert all([not nodes[aid]["is_dummy"] for aid in ancestor_ids])
                assert all([nodes[aid]["type"] in ["head", "list"] for aid in ancestor_ids])
                assert sorted(ancestor_ids) == ancestor_ids
                candid_nodes_for_strutcure.extend(ancestor_ids)
            token_count += new_token_count
            if node['type'] != 'text':
                candid_nodes_for_strutcure.append(node_id)
            else:
                assert node['type'] == 'text'
                if node_id not in candid_nodes_for_text:
                    candid_nodes_for_text[node_id] = set()
                else:
                    assert chunk_id not in candid_nodes_for_text[node_id]
                candid_nodes_for_text[node_id].add(chunk_id)
        else:
            break
    
    assert token_count <= context_len
    assert all([nodes[nid]['type'] != 'text' for nid in candid_nodes_for_strutcure])
    assert all([nodes[nid]['type'] == 'text' for nid in candid_nodes_for_text])

    candid_nodes = sorted(
        list(
            set(candid_nodes_for_strutcure).union(
                set(candid_nodes_for_text.keys())
            )
        )
    )

    context = ""
    for nid in candid_nodes:
        candid_node = nodes[nid]
        assert not candid_node["is_dummy"]
        heading_string = candid_node["heading"]
        if heading_string != "":
            heading_string += "\n\n"

        text_string = ""
        if nid in candid_nodes_for_text:
            chunks = sorted(list(candid_nodes_for_text[nid]))
            for cid in chunks:
                assert cid >= 0 and cid < len(candid_node["texts"])
                chunk_text = candid_node["texts"][cid]
                if chunk_text != "":
                    chunk_text += "\n\n"
                text_string += chunk_text

        context += heading_string + text_string

    assert len(tokenizer.encode(context)) >= token_count
    # assert len(tokenizer.encode(context)) <= context_len
    print(f"actual_context_len = {len(tokenizer.encode(context))}, limit = {context_len}")
    return context

def generate_context(dataset, sht_type):
    sht_config = (
        ("sht", sht_type if sht_type != 'shed' else None, "sbert", True, True, True, 0.2)
    ) 

    index_path = config_utils.get_index_jsonl_path(dataset, config_utils.context_config_to_index_config(sht_config))
    query_path = os.path.join(
        config_utils.DATA_ROOT_FOLDER,
        dataset,
        "queries.json"
    )
    contxt_path = os.path.join(
        os.path.dirname(index_path),
        "exclude_h_for_token_count",
        "context0.2",
        "context.jsonl"
    )
    existing_contxt_ids = set()
    if os.path.exists(contxt_path):
        with open(contxt_path, 'r') as f:
            existing_contxt_ids = set([json.loads(line)['id'] for line in f.readlines()])
    else:
        os.makedirs(os.path.dirname(contxt_path), exist_ok=True)

    with open(index_path, 'r') as f:
        index_info_list = [json.loads(line) for line in f.readlines()]

    with open(query_path, 'r') as f:
        query_info_list = json.load(f)

    assert len(index_info_list) == len(query_info_list)

    for query_info, index_info in zip(query_info_list, index_info_list):
        assert query_info['id'] == index_info['id']
        if query_info['id'] in existing_contxt_ids:
            print(f"skip existing query id {query_info['id']}")
            continue

        filename = query_info['file_name']
        sht_path = os.path.join(
            os.path.dirname(os.path.dirname(index_path)),
            "sht",
            filename + ".json"
        )
        assert os.path.exists(sht_path), sht_path
        with open(sht_path, 'r') as file:
            sht_data = json.load(file)
        
        nodes = sht_data['nodes']
        indexes = index_info['indexes']
        context_len = get_context_len(
            context_ratio=0.2,
            dataset=dataset,
            sht_json_filename=filename,
            min_context_len=150
        )

        context = _generate_context(indexes, nodes, context_len)
        with open(contxt_path, 'a') as f:
            out_info = {
                "id": query_info['id'],
                "context": context
            }
            f.write(json.dumps(out_info) + "\n")


if __name__ == "__main__":
    dataset = "finance"
    for sht_type in [
        "intrinsic",
        "shed",
        "llm_vision",
        "llm_txt",
        "grobid",
        "wide",
        "deep",
    ]:
        generate_context(dataset, sht_type)

