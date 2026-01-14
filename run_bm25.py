import json
from typing import List
import tiktoken
from rank_bm25 import BM25Okapi
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
import os
from structured_rag.utils import get_nondummy_ancestors
from structured_rag import SHTGenerator, SHTGeneratorConfig, get_context_len
from run_raptor import generate_context as raptor_generate_context
import logging
import logging_config
logging.disable(logging.INFO)
from datetime import datetime

def bm25_indexer(chunks: List[str], query: str) -> List[int]:
    '''
    Return a list of the indexes sorted by the decreasing order of bm25 score. The indexes are 0-based. The 0th chunk has index 0. The tokenizer is "cl100k_base".

    Args:
        - chunks (List[str])

        - query (str)

    Return:
        - sorted list of the indexes
    '''
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenized_chunks = [tokenizer.encode(c) for c in chunks]
    tokenized_query = tokenizer.encode(query)

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)
    assert len(scores) == len(chunks)

    sorted_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return sorted_indices

def get_tree(
    dataset,
    name,
    chunk_size,
    summary_len,
    summarization_model,
    is_intrinsic,
    is_baseline,
):
    node_embedding_model = "sbert"
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset)
    if is_intrinsic:
        root_dir = os.path.join(root_dir, "intrinsic")
    if is_baseline:
        root_dir = os.path.join(root_dir, "baselines")
    if not is_baseline:
        tree_path = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", "sht", name+".json")
        with open(tree_path, 'r') as file:
            tree = json.load(file)
        return tree
    else:
        tree_path = os.path.join(root_dir, "raptor_tree", name+".pkl")
        return tree_path

def get_index_path(
    dataset,
    chunk_size,
    summary_len,
    summarization_model,
    distance_metric,
    embed_hierarchy,
    is_intrinsic,
    is_baseline, 
    is_raptor,
):
    node_embedding_model = "bm25"
    query_embedding_model = "bm25"
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset)
    if is_intrinsic:
        root_dir = os.path.join(root_dir, "intrinsic")
    if is_baseline:
        root_dir = os.path.join(root_dir, "baselines")
    if not is_baseline:
        index_path = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}", "index.jsonl")
    else:
        index_path = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}", f"{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", "index.jsonl")
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    return index_path

def get_context_path(
    dataset,
    chunk_size,
    summary_len,
    summarization_model,
    distance_metric,
    embed_hierarchy,
    context_hierarchy,
    context_raw,
    context_len,
    is_intrinsic,
    is_baseline, 
    is_raptor, 
    is_ordered,
):
    index_path = get_index_path(
        dataset,
        chunk_size,
        summary_len,
        summarization_model,
        distance_metric,
        embed_hierarchy,
        is_intrinsic,
        is_baseline, 
        is_raptor,
    )
    if not is_baseline:
        context_path = os.path.join(os.path.dirname(index_path), f"l{int(context_raw)}.h{int(context_hierarchy)}", f"context{context_len}", "context.jsonl")
    else:
        context_path = os.path.join(os.path.dirname(index_path), f"o{int(is_ordered)}", f"context{context_len}", "context.jsonl")

    os.makedirs(os.path.dirname(context_path), exist_ok=True)
    return context_path
    


class CustomQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        raise ValueError("should not achieve this part")

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        pass

    def summarize(self, context, max_tokens=100, stop_sequence=None):
        raise ValueError("should not achieve this part")

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        pass

    def create_embedding(self, text):
        raise ValueError("should not achieve this part")


def index_sht(
        query,
        query_id,
        tree,
        embed_hierarchy,
):
    chunks = []
    m_id_to_pos = dict()

    for node in tree["nodes"]:
        if node["is_dummy"]:
            continue
        node_id = node["id"]
        ancestors = get_nondummy_ancestors(tree["nodes"], node_id)
        if len(ancestors) > 0:
            assert sorted(ancestors) == ancestors
            assert all([(aid >= 0 and aid < len(tree["nodes"]) for aid in ancestors)])
            assert all([not tree["nodes"][aid]["is_dummy"] for aid in ancestors])
            assert all([tree["nodes"][aid]["type"] != "text" for aid in ancestors])
            assert all([tree["nodes"][aid]["heading"] != "" for aid in ancestors])
            assert sorted(ancestors) == ancestors
        ancestor_string = "\n\n".join([tree["nodes"][aid]["heading"] for aid in ancestors])
        if ancestor_string != "":
            ancestor_string += "\n\n"
        
        heading_string = node["heading"]
        if heading_string != "":
            heading_string += "\n\n"

        if node["type"] == "text":
            assert set(node["embeddings"].keys()) == set(["texts", "hybrid"])
            assert node["heading"] == "" and heading_string == ""
        else:
            assert node["type"] in ["head", "list"]
            assert set(node["embeddings"].keys()) == set(["texts", "hybrid", "heading"])
            assert node["heading"] != ""

        for chunk_id, text in enumerate(node["texts"]):
            id = len(chunks)
            m_id_to_pos[id] = {
                "node_id": node["id"],
                "chunk_id": chunk_id,
            }
            if embed_hierarchy:
                chunks.append(ancestor_string + heading_string + text)
            else:
                chunks.append(heading_string + text)

    sorted_indexes = bm25_indexer(chunks, query)

    index_info = {
        "id": query_id,
        "indexes": [m_id_to_pos[i] for i in sorted_indexes]
    }

    return index_info

def index_raptor(
        query,
        query_id,
        tree,
        is_raptor
):
    custom_summarizer = CustomSummarizationModel()
    custom_embedder = CustomEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )

    RA = RetrievalAugmentation(config=custom_RAConfig, tree=tree)

    if not is_raptor:
        nodes = RA.tree.leaf_nodes.values()
    else:
        nodes = RA.tree.all_nodes.values()

    sorted_nodes = sorted(nodes, key=lambda v: v.index)

    assert list(range(len(sorted_nodes))) == [v.index for v in sorted_nodes]

    chunks = [v.text for v in sorted_nodes] 

    indexes = bm25_indexer(chunks, query)

    assert len(indexes) == len(sorted_nodes)
    assert set(indexes) == set(range(len(sorted_nodes)))

    index_info = {
        "id": query_id,
        "indexes": indexes,
    }

    return index_info

def index(
    dataset,
    chunk_size,
    summary_len,
    summarization_model,
    embed_hierarchy,
    distance_metric,
    context_hierarchy,
    context_raw,
    context_len,
    is_intrinsic,
    is_baseline,
    is_raptor,
    is_ordered,
):  
    print("Indexing")
    index_path = get_index_path(
        dataset,
        chunk_size,
        summary_len,
        summarization_model,
        distance_metric,
        embed_hierarchy,
        is_intrinsic,
        is_baseline, 
        is_raptor,
    )

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    
    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries = json.load(file)


    existing_query_ids = list()
    if os.path.exists(index_path):
        with open(index_path, 'r') as file:
            for l in file:
                index_id = (json.loads(l))["id"]
                assert index_id not in existing_query_ids, f"Duplicate query id: {index_id}"
                existing_query_ids.append(index_id)
    

    for query_info in queries:
        query = query_info["query"]
        query_id = query_info["id"]
        name = query_info["file_name"]
        if query_id in existing_query_ids:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]⚠️ Skipping {query_info['id']}")
            continue

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{cur_time}] Processing {query_info['id']}...")
        try:
            tree = get_tree(
                dataset,
                name,
                chunk_size,
                summary_len,
                summarization_model,
                is_intrinsic,
                is_baseline,
            )
            if not is_baseline:
                index_info = index_sht(query, query_id, tree, embed_hierarchy)
            else:
                index_info = index_raptor(query, query_id, tree, is_raptor) 
            with open(index_path, 'a') as file:
                file.write(json.dumps(index_info) + "\n")
            existing_query_ids.append(query_id)
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]✅ Finished processing {query_info['id']} with is_baseline={is_baseline} is_raptor={is_raptor}")
        except Exception as e:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]✅ Finished processing {query_info['id']} with is_baseline={is_baseline} is_raptor={is_raptor}")
            continue


def generate_context(
    dataset,
    chunk_size,
    summary_len,
    summarization_model,
    embed_hierarchy,
    distance_metric,
    context_hierarchy,
    context_raw,
    context_len,
    is_intrinsic,
    is_baseline,
    is_raptor,
    is_ordered,
):
    context_path = get_context_path(
        dataset,
        chunk_size,
        summary_len,
        summarization_model,
        distance_metric,
        embed_hierarchy,
        context_hierarchy,
        context_raw,
        context_len,
        is_intrinsic,
        is_baseline, 
        is_raptor, 
        is_ordered,
    )

    # if os.path.exists(context_path):
    #     logging.debug(f"Context file already existed: {context_path}!")
    #     return

    existing_query_ids = list()
    if os.path.exists(context_path):
        with open(context_path, 'r') as file:
            for l in file:
                context_info = json.loads(l)
                query_id = context_info["id"]
                assert query_id not in existing_query_ids
                existing_query_ids.append(context_info["id"])

    if not is_baseline:
        queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json")
        with open(queries_path, 'r') as file:
            queries = json.load(file)

        index_path = get_index_path(
            dataset,
            chunk_size,
            summary_len,
            summarization_model,
            distance_metric,
            embed_hierarchy,
            is_intrinsic,
            is_baseline, 
            is_raptor,
        )
        indexes = []
        with open(index_path, 'r') as file:
            for l in file:
                indexes.append(json.loads(l))

        assert len(indexes) == len(queries)

        for index_info, query_info in zip(indexes, queries):
            assert index_info["id"] == query_info["id"]
            
            if query_info["id"] in existing_query_ids:
                print(f"\tquery_id {query_id} already has context, skip!")
                continue

            query = query_info["query"]
            query_id = query_info["id"]
            name = query_info["file_name"]
            true_context_len = get_context_len(
                context_ratio=context_len,
                dataset=dataset,
                sht_json_filename=name,
                min_context_len=round(max(chunk_size, summary_len) * 1.5)
            )
            generator_config = SHTGeneratorConfig(
                use_hierarchy=context_hierarchy,
                use_raw_chunks=context_raw,
                context_len=true_context_len
            )
            generator = SHTGenerator(config=generator_config)
            sht = get_tree(
                dataset,
                name,
                chunk_size,
                summary_len,
                summarization_model,
                is_intrinsic,
                is_baseline,
            )
            context = generator.generate(
                candid_indexes=index_info["indexes"],
                nodes=sht["nodes"]
            )
            context_info = {
                "id": query_id,
                "context": context
            }
            assert query_id not in existing_query_ids
            existing_query_ids.append(query_id)
            with open(context_path, 'a') as file:
                file.write(json.dumps(context_info) + "\n")
    else:
        raptor_generate_context(
            dataset=dataset,
            query_embedding_model="bm25",
            is_ordered=is_ordered,
            is_raptor=is_raptor,
            context_len=context_len,
        )



if __name__ == "__main__":
    chunk_size = 100
    summary_len = 100
    summarization_model = "gpt-4o-mini"
    embed_hierarchy = True
    distance_metric = "cosine"
    context_hierarchy = True
    context_raw = True
    context_len = 1000
    is_intrinsic = False
    for is_baseline in [True, False]:
        for is_raptor in [True, False]:
            print(is_baseline, is_raptor)
            if (is_baseline == False and is_raptor == False):
                continue
            dataset = "finance"
            index(
                dataset,
                chunk_size,
                summary_len,
                summarization_model,
                embed_hierarchy,
                distance_metric,
                context_hierarchy,
                context_raw,
                context_len,
                is_intrinsic,
                is_baseline,
                is_raptor,
                is_ordered=None,
            )
            # for is_ordered in [False, True]:
            #     generate_context(
            #         dataset,
            #         chunk_size,
            #         summary_len,
            #         summarization_model,
            #         embed_hierarchy,
            #         distance_metric,
            #         context_hierarchy,
            #         context_raw,
            #         context_len,
            #         is_intrinsic,
            #         is_baseline,
            #         is_raptor,
            #         is_ordered,
            #     )