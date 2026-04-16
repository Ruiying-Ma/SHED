import os

CHUNK_SIZE = 100
SUMMARY_LEN = 100
NODE_EMBEDDING_MODEL_LIST = ["sbert", "dpr", "te3small", "bm25"]
SUMMARIZATION_MODEL = "gpt-4o-mini"
DISTANCE_METRIC = "cosine"
CONTEXT_LEN_RATIO_LIST = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
# CONTEXT_LEN_RATIO_LIST = [0.1, 0.2, 0.3, 0.4]
DATASET_LIST = ["civic", "contract", "qasper", "finance"]
SHT_TYPE_LIST = ["grobid", "intrinsic"]
RAG_METHOD_LIST = ["sht", "raptor", "vanilla"]
DATA_ROOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# config: the tuple that determines the context.jsonl
# config: method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks
CONTEXT_CONFIG_LIST = (
    # end-to-end
    [("vanilla", None, nem, None, None, None, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST] + 
    [("raptor", None, nem, None, None, None, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST] + 
    [("sht", None, nem, True, True, True, r) for nem in NODE_EMBEDDING_MODEL_LIST for r in CONTEXT_LEN_RATIO_LIST] + 
    # ablation on SHT
    [("sht", sht_type, "sbert", True, True, True, 0.2) for sht_type in SHT_TYPE_LIST] + 
    # ablation on HI
    [("sht", None, "sbert", False, True, True, 0.2), ("sht", None, "sbert", True, False, True, 0.2), ("sht", None, "sbert", False, False, True, 0.2)] +
    # ablation on CI
    [("sht", None, "sbert", True, True, False, 0.2)]
)

# print(len(CONTEXT_CONFIG_LIST))

# config: the tuple that determines the index.jsonl
# config: method, sht_type, node_embedding_model, embed_hierarchy
INDEX_CONFIG_LIST = sorted(list(set([
    tuple([i for i in config[:4]])
    for config in CONTEXT_CONFIG_LIST
])), key=lambda t: str(t))
# print("\n".join([str(t) for t in INDEX_CONFIG_LIST]))


def context_config_to_index_config(context_config):
    return tuple([i for i in context_config[:4]])


def get_index_jsonl_path(dataset, index_config_tuple):
    # config: method, sht_type, node_embedding_model, embed_hierarchy
    method, sht_type, node_embedding_model, embed_hierarchy = index_config_tuple

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

def get_config_jsonl_path(root_folder, dataset, context_config_tuple, context_len_ratio):
    index_config_tuple = context_config_to_index_config(context_config_tuple)
    index_jsonl_path = get_index_jsonl_path(root_folder, dataset, index_config_tuple)
    context_folder = os.path.dirname(index_jsonl_path)
    
    method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks = context_config_tuple
        