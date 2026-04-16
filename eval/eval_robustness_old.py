import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
from structured_rag import utils
import tiktoken
import eval.utils as eval_utils
from collections import Counter

TOKENIZER = tiktoken.get_encoding("cl100k_base")
DELIMETER = " "


def normalize_str(s: str):
    return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))

def compare_str(str1: str, str2: str) -> float:
    str1_tokens = normalize_str(str1).split()
    str2_tokens = normalize_str(str2).split()
    common = Counter(str1_tokens) & Counter(str2_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
    precision = num_same / len(str1_tokens)
    recall = num_same / len(str2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def clean_node(node_dict):
    new_node = {'id': node_dict['id']}
    if 'texts' in node_dict:
        new_node['text'] = node_dict['text']
    if 'type' in node_dict:
        new_node['type'] = node_dict['type']
    if 'is_dummy' in node_dict:
        new_node['is_dummy'] = node_dict['is_dummy']
    
    return new_node

# true, grobid, shed, llm
def load_sht(dataset, filename, sht_type):
    tmp_sht_type = sht_type if sht_type != 'shed' else None
    rag_config = ("sht", tmp_sht_type, "sbert", True)
    index_jsonl_path = config.get_index_jsonl_path(dataset, rag_config)
    sht_path = os.path.join(
        os.path.dirname(os.path.dirname(index_jsonl_path)),
        "sht",
        filename + ".json"
    )
    assert os.path.exists(sht_path), sht_path

    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    cleaned_sht_nodes = [clean_node(n) for n in raw_sht_nodes]
    return cleaned_sht_nodes

def get_candid_sht_chunks(target_chunk, sht):
    candid_chunk_id_list = []
    for node in sht:
        if 'text' in node:
            if target_chunk in node['text']:
                candid_chunk_id_list.append(node['id'])

    return candid_chunk_id_list


def get_bottom_up_chunks(m_type_sht):
    assert "intrinsic" in m_type_sht
    chunk_size = 20

    tot_chunk_list = []
    for node in m_type_sht["intrinsic"]:
        if 'text' not in node:
            print(node)
            continue
        node_text = node['text']
        chunk_txt_list = utils.split_text_into_chunks(chunk_size, node_text, TOKENIZER) 
        for chunk_txt in chunk_txt_list:
            chunk_dict = {
                'id': len(tot_chunk_list),
                'text': chunk_txt,
                'intrinsic': [node['id']],
            }
            for sht_type in m_type_sht:
                if sht_type != 'intrinsic':
                    assert sht_type not in chunk_dict
                    chunk_dict[sht_type] = get_candid_sht_chunks(chunk_txt, m_type_sht[sht_type])
            if all([len(chunk_dict[sht_type]) > 0 for sht_type in m_type_sht]):
                tot_chunk_list.append(chunk_dict)

    assert [c['id'] for c in tot_chunk_list] == list(range(len(tot_chunk_list)))
    return tot_chunk_list

def get_ancestor_str(sht, node_id):
    assert node_id in sht, f"node_id = {node_id}, len(sht) = {len(sht)}"
    node = sht[node_id]
    if 'is_dummy' in node:
        assert node['is_dummy'] == False, f"node_id = {node_id}"

    root_to_parent = utils.get_nondummy_ancestors(sht, node_id)
    assert node_id not in root_to_parent
    if 'type' in node and node['type'] != 'Text':
        root_to_parent.append(node_id)
    assert sorted(root_to_parent) == root_to_parent
    assert all([sht[nid]['is_dummy'] == False for nid in root_to_parent])
    assert all(['text' in sht[nid] for nid in root_to_parent])
    ancestor_str = DELIMETER.join([sht[nid]['text'] for nid in root_to_parent])
    return ancestor_str

def eval_bottom_up_chunks(dataset, filename, sht_type_list):
    assert "intrinsic" in sht_type_list

    m_type_sht = {
        sht_type: load_sht(dataset, filename, sht_type)
        for sht_type in sht_type_list
    }

    chunk_list = get_bottom_up_chunks(m_type_sht)

    score_list = []
    for chunk_info in chunk_list:
        intrinsic_id = chunk_info['intrinsic'][0]
        intrinsic_ancestor_str = get_ancestor_str(m_type_sht['intrinsic'], intrinsic_id)
        score_info = {
            'id': chunk_info['id'],
        }
        assert score_info['id'] == len(score_info)
        for sht_type in sht_type_list:
            if sht_type == "intrinsic":
                continue
            assert sht_type in chunk_info
            candid_nodeid_list = chunk_info[sht_type]
            assert len(candid_nodeid_list) > 0
            assert sht_type not in score_info
            score_info[sht_type] = [
                compare_str(intrinsic_ancestor_str, get_ancestor_str(m_type_sht[sht_type], nid))
                for nid in candid_nodeid_list
            ]
            assert len(score_info[sht_type]) == len(chunk_info[sht_type])
        score_list.append(score_info)
    assert [s['id'] for s in score_list] == [c['id'] for c in chunk_list]
    

    return score_list


if __name__ == "__main__":
    score_list = eval_bottom_up_chunks(
        dataset="civic",
        filename="01262022-1835",
        sht_type_list=['intrinsic', 'shed', 'grobid']
    )
    print(score_list)