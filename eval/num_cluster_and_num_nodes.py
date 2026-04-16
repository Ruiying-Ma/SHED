import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import traceback
import logging
import logging_config


def n_cluster_and_n_nodes(dataset):
    sht_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht")
    clustering_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "node_clustering")

    file_name_list = sorted(os.listdir(sht_dir))

    n_cluster_list = []
    n_node_list = []
    tree_depth_list = []


    for file_name in file_name_list:
        sht_path = os.path.join(sht_dir, file_name)
        clustering_path = os.path.join(clustering_dir, file_name)

        with open(sht_path, 'r') as f:
            sht_dict = json.load(f)
            sht = sht_dict['nodes']
        
        with open(clustering_path, 'r') as f:
            clustering = json.load(f)

        n_clusters = len(set([item['cluster_id'] for item in clustering if 'cluster_id' in item]))

        n_cluster_header = len([n for n in clustering if n['type'] in ['Title', 'Section header', 'List item']])

        n_sht_header = len([n for n in sht if n["is_dummy"] == False and n["type"] in ["head", "list"]])

        if len(sht_dict['m_id_to_height'].values()) > 0:
            tree_depth = max(sht_dict['m_id_to_height'].values()) + 1
        else:
            tree_depth = 0

        

        assert n_cluster_header == n_sht_header, f"Mismatch in headers for {file_name}: clustering headers = {n_cluster_header}, SHT headers = {n_sht_header}"

        # ratio = n_clusters / n_cluster_header if n_cluster_header > 0 else None
        # if ratio is not None:
        #     ratio_list.append(ratio)
        n_cluster_list.append(n_clusters)
        n_node_list.append(n_sht_header)
        tree_depth_list.append(tree_depth)

    avg_n_clusters = sum(n_cluster_list) / len(n_cluster_list)
    avg_n_nodes = sum(n_node_list) / len(n_node_list)
    avg_tree_depth = sum(tree_depth_list) / len(tree_depth_list)
    print(f"Dataset: {dataset}")
    print(f"Average number of clusters: {avg_n_clusters:.2f}")
    print(f"Average number of header nodes in SHT: {avg_n_nodes:.2f}")
    print(f"Average tree depth of SHT: {avg_tree_depth:.2f}")
    tot_n_clusters = sum(n_cluster_list)
    tot_n_nodes = sum(n_node_list)
    tot_tree_depth = sum(tree_depth_list)
    max_n_clusters = max(n_cluster_list)
    max_n_nodes = max(n_node_list)
    max_tree_depth = max(tree_depth_list)
    print(f"Max number of clusters in a document: {max_n_clusters}")
    print(f"Max number of header nodes in SHT in a document: {max_n_nodes}")
    print(f"Max tree depth of SHT in a document: {max_tree_depth}")

    return tot_n_clusters, tot_n_nodes, tot_tree_depth

if __name__ == "__main__":
    tot_n_clusters = 0
    tot_n_nodes = 0
    tot_tree_depth = 0
    for dataset in ['civic', 'contract', 'finance', 'qasper']:
        n_cluster, n_node, tree_depth = n_cluster_and_n_nodes(dataset)
        tot_n_clusters += n_cluster
        tot_n_nodes += n_node
        tot_tree_depth += tree_depth

    tot_file_num = 592
    avg_clusters = tot_n_clusters / tot_file_num
    avg_nodes = tot_n_nodes / tot_file_num
    avg_tree_depth = tot_tree_depth / tot_file_num
    print(f"Overall average number of clusters per document: {avg_clusters:.2f}")
    print(f"Overall average number of header nodes in SHT per document: {avg_nodes:.2f}")
    print(f"Overall average tree depth of SHT per document: {avg_tree_depth:.2f}")

        