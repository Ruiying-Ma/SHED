import os

CHUNK_SIZE = 100
SUMMARY_LEN = 100
NODE_EMBEDDING_MODEL_LIST = ["sbert", "dpr", "bm25", "te3small"]
SUMMARIZATION_MODEL = "gpt-4o-mini"
DISTANCE_METRIC = "cosine"
CONTEXT_LEN_RATIO_LIST = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
DATASET_LIST = ["civic", "contract", "qasper", "finance"]
SHT_TYPE_LIST = ["grobid", "intrinsic"]
RAG_METHOD_LIST = ["sht", "raptor", "vanilla"]
DATA_ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CONTEXT_CONFIG_LIST = (
    # end-to-end
    [("graphrag", None, None, None, None, None, None)] + 
    [("hipporag", None, "sbert", None, None, None, r) for r in CONTEXT_LEN_RATIO_LIST] + 
    [("vanilla", None, "sbert", None, None, None, r) for r in CONTEXT_LEN_RATIO_LIST] + 
    [("raptor", None, "sbert", None, None, None, r) for r in CONTEXT_LEN_RATIO_LIST] + 
    [("sht", None, "sbert", True, True, True, r) for r in CONTEXT_LEN_RATIO_LIST] +
    # [("hipporag", None, "sbert", None, None, None, 0.4)] + 
    # [("vanilla", None, "sbert", None, None, None, 0.4)] + 
    # [("raptor", None, "sbert", None, None, None, 0.4)] + 
    # [("sht", None, "sbert", True, True, True, 0.4)]
    ## ablation on embedding models
    [("hipporag", None, nem, None, None, None, 0.2) for nem in NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
    [("vanilla", None, nem, None, None, None, 0.2) for nem in NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
    [("raptor", None, nem, None, None, None, 0.2) for nem in NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
    [("sht", None, nem, True, True, True, 0.2) for nem in NODE_EMBEDDING_MODEL_LIST if nem != "sbert"] + 
    # # # ablation on SHT
    [("sht", "intrinsic", "sbert", True, True, True, 0.2)] +
    # [("sht", "grobid", "sbert", True, True, True, 0.2)] + 
    [("sht", "wide", "sbert", True, True, True, 0.2)] + 
    [("sht", "deep", "sbert", True, True, True, 0.2)] + 
    [("sht", "llm_txt", "sbert", True, True, True, 0.2)] + 
    [("sht", "llm_vision", "sbert", True, True, True, 0.2)] +
    # [("sht", None, "sbert", True, True, True, 0.2)] +
    # # # ablation on HI
    [("sht", None, "sbert", False, True, True, 0.2), ("sht", None, "sbert", True, False, True, 0.2), ("sht", None, "sbert", False, False, True, 0.2)] + 
    # # # ablation on CI
    [("sht", None, "sbert", True, True, False, 0.2)]
)


# config: the tuple that determines the index.jsonl
# config: method, sht_type, node_embedding_model, embed_hierarchy
INDEX_CONFIG_LIST = sorted(list(set([
    tuple([i for i in config[:4]])
    for config in CONTEXT_CONFIG_LIST
])), key=lambda t: str(t))


def context_config_to_index_config(context_config):
    return tuple([i for i in context_config[:4]])


def get_index_jsonl_path(dataset, index_config_tuple):
    # config: method, sht_type, node_embedding_model, embed_hierarchy
    method, sht_type, node_embedding_model, embed_hierarchy = index_config_tuple
    if method in ["graphrag", "hipporag"]:
        raise ValueError(f"{method} doesn't have index jsonl file!")

    query_embedding_model = node_embedding_model

    index_jsonl_path = os.path.join(DATA_ROOT_FOLDER, dataset)
    if method != "sht":
        index_jsonl_path = os.path.join(index_jsonl_path, "baselines")
        index_folder_suffix = f"raptor{int(method=='raptor')}"
    else:
        assert embed_hierarchy != None
        assert isinstance(embed_hierarchy, bool)
        index_folder_suffix = f"h{int(embed_hierarchy == True)}"

    if sht_type != None:
        index_jsonl_path = os.path.join(index_jsonl_path, sht_type)
    
    index_jsonl_path = os.path.join(
        index_jsonl_path,
        f"{node_embedding_model}.{SUMMARIZATION_MODEL}.c{CHUNK_SIZE}.s{SUMMARY_LEN}",
        f"{query_embedding_model}.{DISTANCE_METRIC}.{index_folder_suffix}",
        "index.jsonl"
    )
    
    assert os.path.exists(index_jsonl_path), index_jsonl_path
    return index_jsonl_path

def get_config_jsonl_path(dataset, context_config_tuple):
    if context_config_tuple[0] == "hipporag":
        context_len_ratio = context_config_tuple[-1]
        node_embedding_model = context_config_tuple[2]
        context_jsonl_path = os.path.join(DATA_ROOT_FOLDER, dataset, "baselines", f"{node_embedding_model}.{SUMMARIZATION_MODEL}.c{CHUNK_SIZE}.s{SUMMARY_LEN}", f"{node_embedding_model}.{DISTANCE_METRIC}.hipporag", "o0", f"context{context_len_ratio}", "context.jsonl")
        return context_jsonl_path
    
    index_config_tuple = context_config_to_index_config(context_config_tuple)
    index_jsonl_path = get_index_jsonl_path(dataset, index_config_tuple)
    context_folder = os.path.dirname(index_jsonl_path)
    
    method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks, context_len_ratio = context_config_tuple
    
    context_jsonl_path = os.path.join(
        context_folder,
        "o0" if method != "sht" else f"l{int(use_raw_chunks == True)}.h{int(context_hierarchy == True)}",
        f"context{context_len_ratio}",
        "context.jsonl"
    )

    return context_jsonl_path
