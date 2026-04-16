import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import os
import json
import tiktoken
from structured_rag.utils import get_context_len

TOKENIZER = tiktoken.get_encoding("cl100k_base")

m_dataset_size = {
    "civic": 418,
    "contract": 1241,
    "qasper": 1451,
    "finance": 150,
}

# for dataset in ["civic", "contract", "qasper"]:
for dataset in ["finance"]:
    for context_config in config.CONTEXT_CONFIG_LIST:
        print(dataset, context_config)
        method, sht_type, node_embedding_model, embed_hierarchy, context_hierarchy, use_raw_chunks, context_len_ratio = context_config
        # context_jsonl_path
        context_jsonl_path = config.get_config_jsonl_path(dataset, context_config)
        assert not os.path.exists(context_jsonl_path)
        os.makedirs(os.path.dirname(context_jsonl_path), exist_ok=True)
        # index_jsonl_path
        index_jsonl_path = f"/home/ruiying/SHTRAG/hipporag/query_solution/{dataset}/{node_embedding_model}.jsonl"
        assert os.path.exists(index_jsonl_path)
        with open(index_jsonl_path, 'r') as file:
            raw_index_list = [json.loads(l) for l in file.readlines()]
        sorted_index_list = sorted(raw_index_list, key=lambda r: r['id'])
        assert list([ind['id'] for ind in sorted_index_list]) == list(range(m_dataset_size[dataset]))
        # queries
        query_jsonl_path = f"/home/ruiying/SHTRAG/data/{dataset}/queries.json"
        with open(query_jsonl_path, 'r') as file:
            query_info_list = json.load(file)
        assert [qi['id'] for qi in query_info_list] == list(range(m_dataset_size[dataset]))
        # gen context
        for index_info, query_info in zip(sorted_index_list, query_info_list):
            true_context_len = get_context_len(
                context_ratio=context_len_ratio,
                dataset=dataset,
                sht_json_filename=query_info['file_name'],
                min_context_len=150
            )
            context = ""
            context_token_count = 0
            for chunk in index_info["docs"]:
                text = chunk + "\n\n"
                text_token_count = len(TOKENIZER.encode(text))
                if context_token_count + text_token_count <= true_context_len:
                    context += text
                    context_token_count += text_token_count
                else:
                    break
            assert len(TOKENIZER.encode(context)) == context_token_count
            context_info = {
                "id": query_info['id'],
                "context": context,
            }
            with open(context_jsonl_path, 'a') as file:
                file.write(json.dumps(context_info) + '\n')
            


