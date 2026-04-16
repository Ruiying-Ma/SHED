import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from structured_rag import StructuredRAG
# from script_python.get_doc_tokens import count_tokens_from_raptor_leaves
from config import CONTEXT_CONFIG_LIST, CHUNK_SIZE, SUMMARY_LEN, SUMMARIZATION_MODEL, DISTANCE_METRIC, get_index_jsonl_path, get_config_jsonl_path
import json
import tiktoken
import colorlog
import logging
import logging_config
from run_bm25 import generate_context as bm25_generate_context
from run_raptor import generate_context as raptor_generate_context
import argparse

SAFETY_CHECK = True

def collect_orig_existing_index_jsonl_path():
    data_root_folder = "/home/ruiying/SHTRAG/data"
    root = Path(data_root_folder)
    return [str(p) for p in root.rglob("index.jsonl") if "/example/" not in p]

def build_context(dataset):
    for context_config in CONTEXT_CONFIG_LIST:
        if SAFETY_CHECK == True:
            context_path = get_config_jsonl_path(dataset, context_config)
            assert not os.path.exists(context_path), f"{context_config} ERROR: {context_path} already existed!"
            continue

        method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks, context_len_ratio = context_config
        print(context_config)
        
        query_embedding_model = node_embedding_model
        
        index_jsonl_path = get_index_jsonl_path(
            dataset=dataset,
            index_config_tuple=(method, sht_type, node_embedding_model, embed_hierarchy)
        )
        queries_path = os.path.join("/home/ruiying/SHTRAG/data", dataset, "queries.json")
        with open(queries_path, 'r') as file:
            queries_info = json.load(file)

        print(f"Building context for {dataset} with config: {context_config}")

        if node_embedding_model != "bm25" and method == "sht":
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(index_jsonl_path)))
            for qid, query_info in enumerate(queries_info):
                assert qid == query_info["id"]
                rag = StructuredRAG(
                    root_dir=root_dir,
                    chunk_size=CHUNK_SIZE,
                    summary_len=SUMMARY_LEN,
                    node_embedding_model=node_embedding_model,
                    query_embedding_model=query_embedding_model,
                    summarization_model=SUMMARIZATION_MODEL,
                    embed_hierarchy=embed_hierarchy,
                    distance_metric=DISTANCE_METRIC,
                    context_hierarchy=context_hierarchy,
                    context_raw=use_raw_chunks,
                    context_len=context_len_ratio
                )
                rag.generate_context(
                    name=query_info["file_name"],
                    query=query_info["query"],
                    query_id=qid,
                )
        elif node_embedding_model != "bm25" and method != "sht":
            raptor_generate_context(
                dataset=dataset,
                query_embedding_model=query_embedding_model,
                is_ordered=False,
                is_raptor=(method == "raptor"),
                context_len=context_len_ratio,
            )
        else:
            assert node_embedding_model == "bm25"
            bm25_generate_context(
                dataset=dataset,
                chunk_size=CHUNK_SIZE,
                summary_len=SUMMARY_LEN,
                summarization_model=SUMMARIZATION_MODEL,
                embed_hierarchy=embed_hierarchy,
                distance_metric=DISTANCE_METRIC,
                context_hierarchy=context_hierarchy,
                context_raw=use_raw_chunks,
                context_len=context_len_ratio,
                is_intrinsic=(sht_type == "intrinsic"),
                is_baseline=(method != "sht"),
                is_raptor=(method == "raptor"),
                is_ordered=False,
            )



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--safety-check", action='store_true', help="Whether to check the context configures in config.py before continue")
    args = parser.parse_args()
    SAFETY_CHECK = bool(args.safety_check)

    build_context("civic")
    build_context("contract")
    build_context("qasper")
    build_context("finance")