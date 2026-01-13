import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import traceback
import logging
import logging_config


def n_cluster_n_nodes(dataset):
    sht_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht")
    clustering_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "node_clustering")

    file_name_list = sorted(os.listdir(sht_dir))

    ratio_list = []

    for file_name in file_name_list:
        sht_path = os.path.join(sht_dir, file_name)
        clustering_path = os.path.join(clustering_dir, file_name)

        with open(sht_path, 'r') as f:
            sht = json.load(f)['nodes']
        
        with open(clustering_path, 'r') as f:
            clustering = json.load(f)

        n_clusters = len(set([item['cluster_id'] for item in clustering if 'cluster_id' in item]))

        n_cluster_header = len([n for n in clustering if n['type'] in ['Title', 'Section header', 'List item']])

        n_sht_header = len([n for n in sht if n["is_dummy"] == False and n["type"] in ["head", "list"]])

        assert n_cluster_header == n_sht_header, f"Mismatch in headers for {file_name}: clustering headers = {n_cluster_header}, SHT headers = {n_sht_header}"

        ratio = n_clusters / n_cluster_header if n_cluster_header > 0 else None
        if ratio is not None:
            ratio_list.append(ratio)

    avg_ratio = sum(ratio_list) / len(ratio_list)

    print(f"Average ratio of number of clusters to number of header nodes in SHT for dataset {dataset}: {avg_ratio:.4f}")

if __name__ == "__main__":
    for dataset in ['civic', 'contract', 'finance', 'qasper']:
        n_cluster_n_nodes(dataset)

        