import os
import json

def load_hard_queries(dataset):
    query_path = f"/home/ruiying/SHTRAG/script_python/easy_queries_from_true_sht_{dataset}.jsonl"
    query_info_list = []
    with open(query_path, 'r') as file:
        for f in file:
            query_info_list.append(json.loads(f))

    return [
        q['id']
        for q in query_info_list
        if q['hard_level'] in {0} and len(q['clusters']) > 0
    ]


print(len(load_hard_queries('contract')))
