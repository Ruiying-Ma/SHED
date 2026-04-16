import json
import os

def count_queries_with_hier_prov(dataset):
    prov_info_path = f"/home/ruiying/SHTRAG/script_python/easy_queries_{dataset}.jsonl"
    queries_path = f"/home/ruiying/SHTRAG/data/{dataset}/queries.json"

    tot_query_list = []
    with open(queries_path, 'r') as file:
        queries = json.load(file)
        if dataset == "contract":
            tot_query_list = [q for q in queries if q['answer'] != "NotMentioned"]
        else:
            tot_query_list = [q for q in queries]
    
    tot_query_ids = set([q['id'] for q in tot_query_list])

    hier_prov_query_ids = []
    with open(prov_info_path, 'r') as file:
        for line in file:
            prov_info = json.loads(line)
            if prov_info['id'] not in tot_query_ids:
                continue
            use_hier = False
            for prov in prov_info['prov']:
                for node in prov['matched_nodes']:
                    if node['type'] != 'text':
                        use_hier = True
                        break
                if use_hier == True:
                    break
            if use_hier == True:
                hier_prov_query_ids.append(prov_info['id'])
    
    assert len(set(hier_prov_query_ids)) == len(hier_prov_query_ids)

    return len(hier_prov_query_ids) / len(tot_query_list)


def count_queries_with_only_hier_prov(dataset):
    prov_info_path = f"/home/ruiying/SHTRAG/script_python/easy_queries_{dataset}.jsonl"
    queries_path = f"/home/ruiying/SHTRAG/data/{dataset}/queries.json"

    tot_query_list = []
    with open(queries_path, 'r') as file:
        queries = json.load(file)
        if dataset == "contract":
            tot_query_list = [q for q in queries if q['answer'] != "NotMentioned"]
        else:
            tot_query_list = [q for q in queries]
    
    tot_query_ids = set([q['id'] for q in tot_query_list])

    hier_prov_query_ids = []
    with open(prov_info_path, 'r') as file:
        for line in file:
            prov_info = json.loads(line)
            if prov_info['id'] not in tot_query_ids:
                continue
            if len(prov_info['prov']) == 0:
                continue
            use_hier = True
            for prov in prov_info['prov']:
                for node in prov['matched_nodes']:
                    if node['type'] == 'text':
                        use_hier = False
                        break
                if use_hier == False:
                    break
            if use_hier == True:
                hier_prov_query_ids.append(prov_info['id'])
    
    assert len(set(hier_prov_query_ids)) == len(hier_prov_query_ids)

    return len(hier_prov_query_ids) / len(tot_query_list)


if __name__ == "__main__":
    datasets = ["contract", "qasper", "finance"]
    for dataset in datasets:
        ratio = count_queries_with_only_hier_prov(dataset)
        print(f"Dataset: {dataset}, Ratio of queries using hierarchical provenance: {ratio:.4f}")

            