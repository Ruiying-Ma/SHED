import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import traceback
import logging
import logging_config
from structured_rag import utils
import eval.utils as eval_utils
from collections import Counter

# each header node in the true SHT has info:cluster_id

HEADER_TYPES = ["head", "list"]
# HEADER_TYPES = ["head"]

DATA_ROOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TEXT_CLUSTER_ID = -100

# def _clean_txt(txt: str):
#     return (" ".join(txt.split())).lower().strip()

def _normalize_str(s: str):
    return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))
    # return eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s)))

def _sim_txt(t1: str, t2: str):
    t1_tokens = t1.split()
    t2_tokens = t2.split()

    common = Counter(t1_tokens) & Counter(t2_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0
    precision = num_same / len(t1_tokens)
    recall = num_same / len(t2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    if recall == 1:
        return 1, f1
    if precision == 1:
        return 1, f1
    return f1, f1

def load_headers(dataset, filename):
    """
    Map each true header to its cluster ID from SHED
    """
    logging.info(f"Loading and mapping headers for {filename} in dataset {dataset}...")
    true_sht_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        "intrinsic",
        f"sbert.gpt-4o-mini.c100.s100",
        "sht",
        filename + ".json"
    )
    node_clustering_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        "node_clustering",
        filename + ".json"
    )
    assert os.path.exists(true_sht_path), true_sht_path
    assert os.path.exists(node_clustering_path), node_clustering_path
    with open(true_sht_path, 'r') as file:
        sht = json.load(file)['nodes']
    with open(node_clustering_path, 'r') as file:
        node_clustering = json.load(file)

    header_sht_nodes = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    cluster_header_nodes = [n for n in node_clustering if 'cluster_id' in n]

    for hnode in header_sht_nodes:
        assert 'clean_text' not in hnode
        hnode['clean_text'] = _normalize_str(hnode['heading'])
    for cnode in cluster_header_nodes:
        assert 'clean_text' not in cnode
        assert cnode['type'] in ["Title", "Section header", "List item"]
        cnode['clean_text'] = _normalize_str(cnode['text'])

    
    for hnode in header_sht_nodes:
        if hnode['type'] == "list":
            matched_type_cluster_header_nodes = [n for n in cluster_header_nodes if n['type'] == "List item"]
        else:
            matched_type_cluster_header_nodes = [n for n in cluster_header_nodes if n['type'] in ["Title", "Section header"]]
        if len(matched_type_cluster_header_nodes) == 0:
            hnode['inferred_cluster'] = TEXT_CLUSTER_ID
            continue
        matched_similarity = [_sim_txt(hnode['clean_text'], cnode['clean_text']) for cnode in matched_type_cluster_header_nodes]
        max_sim = max(matched_similarity)
        candidate_indices = [i for i, sim in enumerate(matched_similarity) if sim == max_sim]
        if len(candidate_indices) > 1:
            logging.warning(f"[{filename}] Multiple candidate clusters for header node {hnode['id']}: {[matched_type_cluster_header_nodes[i] for i in candidate_indices]}")
        best_cnode = matched_type_cluster_header_nodes[candidate_indices[0]]
        hnode['inferred_cluster'] = best_cnode['cluster_id']

    dst_sht_path = os.path.join(
        DATA_ROOT_FOLDER,
        dataset,
        "intrinsic",
        "sbert.gpt-4o-mini.c100.s100",
        "sht_clustering",
        filename + ".json"
    )

    new_sht = []
    for n in sht:
        if 'embeddings' in n:
            del n['embeddings']
        new_sht.append(n)

    os.makedirs(os.path.dirname(dst_sht_path), exist_ok=True)
    with open(dst_sht_path, 'w') as file:
        json.dump({'nodes': new_sht}, file, indent=2)
        







def get_node_prefix(node, sht):
    """Get the prefix of a node, defined as the sequence of header texts from root to the node."""
    ancestors_ids = utils.get_nondummy_ancestors(sht, node['id'])
    ancestors = [sht[aid] for aid in ancestors_ids]
    assert all([h['type'] in HEADER_TYPES for h in ancestors])
    prefix = [h['inferred_cluster'] for h in ancestors if h['type'] in HEADER_TYPES]
    return tuple(prefix)



def is_well_formatted(sht):
    """Each cluster is prefix-unique; Siblings belong to the same cluster"""
    m_cluster_headers = dict()
    headers = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    for h in headers:
        cluster_id = h['inferred_cluster']
        if cluster_id not in m_cluster_headers:
            m_cluster_headers[cluster_id] = []
        m_cluster_headers[cluster_id].append(h)
    
    for cluster_id, header_list in m_cluster_headers.items():
        # unique prefixes
        prefixes = set()
        for h in header_list:
            prefix = get_node_prefix(h, sht)
            prefixes.add(prefix)
        # logging.info(f"Cluster ID {cluster_id} has prefixes: {prefixes}")
        if len(prefixes) > 1:
            return False
    
    for h in headers:
        for o_h in headers:
            if h['id'] == o_h['id']:
                continue
            if h['nondummy_parent'] != o_h['nondummy_parent']:
                continue
            if h['inferred_cluster'] != o_h['inferred_cluster']:
                return False
        
    return True



def is_loosely_formatted(sht):
    m_cluster_headers = dict()
    headers = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    for h in headers:
        cluster_id = h['inferred_cluster']
        if cluster_id not in m_cluster_headers:
            m_cluster_headers[cluster_id] = []
        m_cluster_headers[cluster_id].append(h)

    for h in headers:
        cluster_id = h['inferred_cluster']
        parent_id = h['nondummy_parent']
        if parent_id == -1:
            continue
        else:
            p_cluster = sht[parent_id]['inferred_cluster']

        indexing_node_id = min([n['id'] for n in m_cluster_headers[cluster_id]])
        assert indexing_node_id <= h['id'], (indexing_node_id, h['id'])
        ancestor_ids = utils.get_nondummy_ancestors(sht, indexing_node_id)
        ancestor_cluster_ids = [sht[aid]['inferred_cluster'] for aid in ancestor_ids]
        if p_cluster not in ancestor_cluster_ids:
            return False
    return True
    

def is_depth_aligned(sht):
    m_cluster_headers = dict()
    headers = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    if len(headers) == 0:
        return True
    for h in headers:
        cluster_id = h['inferred_cluster']
        if cluster_id not in m_cluster_headers:
            m_cluster_headers[cluster_id] = []
        m_cluster_headers[cluster_id].append(h)
    
    for cluster_id, header_list in m_cluster_headers.items():
        # unique levels
        levels = set()
        for h in header_list:
            level = len(utils.get_nondummy_ancestors(sht, h['id']))
            levels.add(level)
        if len(levels) > 1:
            return False
    
    leftmost_child_root = min([h['id'] for h in headers])
    # leftmost child
    for h in headers:
        parent_id = h['nondummy_parent']
        if parent_id == -1:
            if h['id'] != leftmost_child_root:
                return False
            else:
                continue
        parent_node = sht[parent_id]
        nondummy_children = [cid for cid in parent_node['nondummy_children'] if sht[cid]['is_dummy'] == False and sht[cid]['type'] in HEADER_TYPES]
        assert h['id'] in nondummy_children
        leftmost_nondummy_child_id = min(nondummy_children)
        leftmost_child = sht[leftmost_nondummy_child_id]
        assert leftmost_child['id'] == leftmost_nondummy_child_id
        assert leftmost_child['is_dummy'] == False
        assert leftmost_child['type'] in HEADER_TYPES
        if h['id'] != leftmost_child['id']:
            return False
        
    return True


def is_local_first(sht):
    # set semantic depth
    raw_headers = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    headers = sorted(raw_headers, key=lambda x: x['id'])
    m_cluster_sd = dict()
    for h_idx, h in enumerate(headers):
        cluster_id = h['inferred_cluster']
        if cluster_id in m_cluster_sd:
            continue
        if h_idx == 0:
            sd = 0
        else:
            prev_h = headers[h_idx - 1]
            prev_cluster_id = prev_h['inferred_cluster']
            assert prev_cluster_id in m_cluster_sd
            prev_sd = m_cluster_sd[prev_cluster_id]
            sd = prev_sd + 1
        m_cluster_sd[cluster_id] = sd

    for h in headers:
        parent_id = h['nondummy_parent']
        if parent_id == -1:
            continue
        parent = sht[parent_id]
        h_sd = m_cluster_sd[h['inferred_cluster']]
        p_sd = m_cluster_sd[parent['inferred_cluster']]
        if p_sd >= h_sd:
            return False
    return True

def is_global_first(sht):
    # set semantic depth
    raw_headers = [n for n in sht if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    headers = sorted(raw_headers, key=lambda x: x['id'])
    m_cluster_sd = dict()
    for h_idx, h in enumerate(headers):
        cluster_id = h['inferred_cluster']
        if cluster_id in m_cluster_sd:
            continue
        else:
            m_cluster_sd[cluster_id] = cluster_id

    for h in headers:
        parent_id = h['nondummy_parent']
        if parent_id == -1:
            continue
        parent = sht[parent_id]
        h_sd = m_cluster_sd[h['inferred_cluster']]
        p_sd = m_cluster_sd[parent['inferred_cluster']]
        if p_sd >= h_sd:
            return False
    return True



def doc_classification(dataset: str, class_name: str):
    classification_results = []
    doc_cnt = 0
    for pdf_name in sorted(os.listdir(os.path.join(DATA_ROOT_FOLDER, dataset, 'pdf'))):
        file_name = pdf_name.replace(".pdf", "")
        true_sht_path = os.path.join(
            DATA_ROOT_FOLDER,
            dataset,
            "intrinsic",
            f"sbert.gpt-4o-mini.c100.s100",
            "sht",
            file_name + ".json"
        )
        visual_pattern_path = os.path.join(
            DATA_ROOT_FOLDER,
            dataset,
            "intrinsic",
            "visual_patterns",
            file_name + ".json"
        )

        assert os.path.exists(true_sht_path), true_sht_path

        with open(true_sht_path, 'r') as file:
            sht_nodes = json.load(file)['nodes']

        with open(visual_pattern_path, 'r') as file:
            visual_patterns = json.load(file)

        sht_headers = [n for n in sht_nodes if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
        
        if len(sht_headers) != len(visual_patterns):
            logging.warning(f"{file_name}: Mismatch in number of headers and visual patterns: {len(sht_headers)} vs {len(visual_patterns)}")
            continue

        doc_cnt += 1
        assert len(sht_headers) == len(visual_patterns)
        
        for h_node, v_pattern in zip(sht_headers, visual_patterns):
            h_node['inferred_cluster'] = v_pattern['cluster_id']

        if class_name == "well_formatted":
            result = is_well_formatted(sht_nodes)
        elif class_name == "loosely_formatted":
            result = is_loosely_formatted(sht_nodes)
        elif class_name == "depth_aligned":
            result = is_depth_aligned(sht_nodes)
        elif class_name == "local_first":
            result = is_local_first(sht_nodes)
        elif class_name == "global_first":
            result = is_global_first(sht_nodes)
        else:
            raise NotImplementedError(class_name)
        
        assert result in [True, False]
        classification_results.append(result)

    # Save results
    assert len(classification_results) <= len(os.listdir(os.path.join(DATA_ROOT_FOLDER, dataset, 'pdf')))
    assert doc_cnt == len(classification_results)
    num_true = len([res for res in classification_results if res == True])
    print(f"Dataset: {dataset}, Class: {class_name}, Num: {num_true}/{len(classification_results)}, Ratio: {num_true/len(classification_results):.4f}")
    return classification_results


if __name__ == "__main__":
    for dataset in [
        'civic',
        'contract',
        'finance',
        'qasper'
    ]:
        # for pdf_name in sorted(os.listdir(os.path.join(DATA_ROOT_FOLDER, dataset, 'pdf'))):
        #     file_name = pdf_name.replace(".pdf", "")
        #     load_headers(dataset, file_name)
        for class_name in [
            'well_formatted',
            'loosely_formatted',
            'depth_aligned',
            'local_first',
            'global_first'
        ]:
            doc_classification(dataset, class_name)


        

        