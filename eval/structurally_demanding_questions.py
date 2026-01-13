import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging_config
import numpy as np
from structured_rag import split_text_into_sentences, get_nondummy_ancestors
import eval.utils
from collections import Counter

def _clean_txt(txt: str):
    return ("".join(txt.split())).lower().strip()

def _clean_node(node_dict):
    new_node = {
        'id': node_dict['id'],
        'is_dummy': node_dict['is_dummy'],
        'nondummy_parent': node_dict['nondummy_parent']
    }
    if node_dict['is_dummy'] == False:
        new_node['type'] = node_dict['type']
        if node_dict['type'] == 'text':
            assert node_dict['heading'] == ""
            # new_node['texts'] = node_dict['texts']
            # new_node['texts'] = [eval_utils.white_space_fix(t).strip() for t in node_dict['texts']]
            new_node['clean_text'] = "".join([_clean_txt(t).strip() for t in node_dict['texts']])
        else:
            assert node_dict['type'] in ['head', 'list']
            assert len(node_dict['texts']) == 1
            # new_node['texts'] = [node_dict['heading']]
            # new_node['texts'] = [eval_utils.white_space_fix(node_dict['heading']).strip()]
            new_node['clean_text'] = _clean_txt(node_dict['heading'])
    
    return new_node


def _load_true_sht(dataset, filename):
    sht_type = "intrinsic"
    sht_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset,
        sht_type,
        f"sbert.gpt-4o-mini.c100.s100",
        "sht",
        filename + ".json"
    )
    assert os.path.exists(sht_path), sht_path
    # logging.info(f"Loading SHT for {filename} ({sht_type}) from {sht_path}...")
    
    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    cleaned_sht_nodes = [_clean_node(n) for n in raw_sht_nodes]
    return cleaned_sht_nodes


def _map_cluster_to_nodes(cluster, sht_nodes):
    cluster_type = cluster['type']
    candid_node_list = []
    for node in sht_nodes:
        if node['is_dummy'] == True:
            continue
        if node['type'] != cluster['type']:
            continue
        if cluster_type == 'head':
            if node['clean_text'] == cluster['clean_text']:
                candid_node_list.append(node)
        elif cluster_type == 'list':
            if node['clean_text'] == cluster['clean_text']:
                candid_node_list.append(node)
            elif node['clean_text'] in cluster['clean_text']:
                next_nondummy_node_id = node['id'] + 1
                while sht_nodes[next_nondummy_node_id]['is_dummy'] == True:
                    next_nondummy_node_id += 1
                if next_nondummy_node_id < len(sht_nodes):
                    next_nondummy_node = sht_nodes[next_nondummy_node_id]
                    if next_nondummy_node['type'] == 'text':
                        combined_text = node['clean_text'] + next_nondummy_node['clean_text']
                        if cluster['clean_text'] in combined_text:
                            candid_node_list.append(node)
        else:
            assert cluster_type == 'text'
            if cluster['clean_text'] in node['clean_text']:
                candid_node_list.append(node)

    if len(candid_node_list) != 1:
        candid_node_id = int(input(f"Cluster: {cluster}\n Candid nodes: {candid_node_list}"))
        print(f"\t Input node ID: {candid_node_id}")
        candid_node_list = [n for n in sht_nodes if n['id'] == candid_node_id and n['is_dummy'] == False]
    
    assert len(candid_node_list) == 1, (cluster, candid_node_list)
    return candid_node_list[0]


def is_easy(clusters, m_cluster_node, all_nodes, root_cluster_id):
    if len(clusters) == 0:
        return 0
    cluster_ids = sorted([c['id'] for c in clusters])
    assert [c['id'] for c in clusters] == cluster_ids, cluster_ids
    
    chunk_id = -1
    m_cluster_chunk = dict()
    for cluster_id in cluster_ids:
        if cluster_id - 1 in cluster_ids:
            # seq, same chunk
            m_cluster_chunk[cluster_id] = chunk_id
        else:
            # new chunk
            chunk_id += 1
            m_cluster_chunk[cluster_id] = chunk_id

    if chunk_id == 0:
        # one-point provenance
        assert cluster_ids == list(range(min(cluster_ids), max(cluster_ids)+1))
        return 1
    
    # multi-point provenance
    header_cluster_ids = [c['id'] for c in clusters if c['type'] != 'text']
    text_headers = [c['parent'] for c in clusters if c['type'] == 'text']
    if set(header_cluster_ids).issubset(set(text_headers + [root_cluster_id])):
        return 2
    
    is_easy = True
    for cluster_id_1 in cluster_ids:
        if is_easy == False:
            break
        cluster_1 = [c for c in clusters if c['id'] == cluster_id_1][0]
        if cluster_1['type'] == 'text':
            continue

        for cluster_id_2 in cluster_ids:
            print(f"\t[Level 3] Checking clusters: {cluster_id_1}, {cluster_id_2}")
            if is_easy == False:
                break
            if cluster_id_1 >= cluster_id_2:
                continue
            if m_cluster_chunk[cluster_id_1] == m_cluster_chunk[cluster_id_2]:
                print(f"\t\t[Level 3] Same chunk: {cluster_id_1}, {cluster_id_2}")
                continue
            cluster_2 = [c for c in clusters if c['id'] == cluster_id_2][0]
            if cluster_2['type'] == 'text':
                continue
            
            node_id_1 = m_cluster_node[cluster_id_1]
            node_id_2 = m_cluster_node[cluster_id_2]
            node_2_ancestors = get_nondummy_ancestors(all_nodes, node_id_2)
            assert sorted(node_2_ancestors) == node_2_ancestors
            print(f"\t\t[Level 3] node_2_ancestors: {node_2_ancestors}, node_id_1: {node_id_1}")
            if (len(node_2_ancestors) > 0) and (node_id_1 in node_2_ancestors):
                if cluster_2['type'] == 'text':
                    if node_id_1 < node_2_ancestors[-1]:
                        is_easy = False
                        break
                else:
                    is_easy = False
                    break
                
    if is_easy == True:
        return 3
    
    return 0


if __name__ == "__main__":
    hard_levels = {0}
    for dataset in ["contract", "qasper", "finance"]:
        dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_demanding_questions", f"{dataset}.jsonl")
        existing_query_ids = dict()
        if os.path.exists(dst_path):
            with open(dst_path, 'r') as file:
                for line in file:
                    json_line = json.loads(line)
                    assert json_line['id'] not in existing_query_ids, json_line['id']
                    existing_query_ids[json_line['id']] = json_line


        root_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            dataset,
        )

        queries_path = os.path.join(
            root_folder,
            "queries.json"
        )

        with open(queries_path, 'r') as file:
            queries_info = json.load(file)

        prov_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "true_prov", f"{dataset}.jsonl")
        with open(prov_path, 'r') as file:
            prov_info_list = [json.loads(line) for line in file.readlines()]

        assert len(queries_info) == len(prov_info_list), (len(queries_info), len(prov_info_list))

        hard_query_ids = []
        query_cnt = 0
        for query_info, prov_info in zip(queries_info, prov_info_list):
            if dataset == "contract":
                if query_info['answer'] == "NotMentioned":
                    continue
            query_cnt += 1
            assert query_info['id'] == prov_info['id'], (query_info['id'], prov_info['id'])
            if query_info['id'] in existing_query_ids:
                logging.info(f"Skipping query ID: {query_info['id']} (already processed)...")
                existing_q = existing_query_ids[query_info['id']]
                if existing_q['hard_level'] in hard_levels and len(existing_q['clusters']) > 0:
                    hard_query_ids.append(query_info['id'])
                continue

            filename = query_info['file_name']
            logging.info(f"Processing query ID: {query_info['id']} ({filename})...")
            clustering_path = os.path.join(
                "/home/ruiying/SHTRAG/data",
                dataset,
                "intrinsic",
                "node_clustering",
                filename + ".json"
            )
            with open(clustering_path, 'r') as file:
                clustering_info = json.load(file)
            header_cluster_list = [c['id'] for c in clustering_info if c['type'] in ['Title', 'Section header', 'List item']]
            root_cluster_id = min(header_cluster_list) if len(header_cluster_list) > 0 else None

            sht_nodes = _load_true_sht(dataset, filename)

            clusters = []
            existing_ids = set()
            for prov in prov_info['prov']:
                for c in prov['matched_nodes']:
                    if c['id'] not in existing_ids:
                        clusters.append(c)
                        existing_ids.add(c['id'])
            clusters = sorted(clusters, key=lambda x: x['id'])
            
            
            for c in clusters:
                c['clean_text'] = _clean_txt(c['clean_text'])
            
            if len(clusters) == 0:
                easy_info = {
                    'id': query_info['id'],
                    'file_name': filename,
                    'hard_level': 0, # but not counted as structurally demanding
                    'clusters': clusters,
                    'm_cluster_node': dict()
                }
                with open(dst_path, 'a') as file:
                    file.write(json.dumps(easy_info) + "\n")
                continue

            m_cluster_node = dict()
            for cluster in clusters:
                candid_node = _map_cluster_to_nodes(cluster, sht_nodes)
                m_cluster_node[cluster['id']] = candid_node['id']

            hard_level = is_easy(clusters, m_cluster_node, sht_nodes, root_cluster_id)

            easy_info = {
                'id': query_info['id'],
                'file_name': filename,
                'hard_level': hard_level,
                'clusters': clusters,
                'm_cluster_node': m_cluster_node
            }
            with open(dst_path, 'a') as file:
                file.write(json.dumps(easy_info) + "\n")


            if hard_level in hard_levels:
                hard_query_ids.append(query_info['id'])
        
        print(f"Structurally demanding query IDs in {dataset}: {len(hard_query_ids) * 100/query_cnt:.1f}%")