import os
import json

def parse_batch(src_jsonl):
    result_list = []
    with open(src_jsonl, 'r') as src_file:
        for line in src_file:
            job = json.loads(line)
            response_content = job['response']['body']['choices'][0]['message']['content']
            id_str = job['custom_id']
            if "-" not in id_str:
                result_id = int(id_str)
            else:
                job_id = int(id_str.split("-")[0])
                chunk_idx = int(id_str.split("-")[1])
                result_id = (job_id, chunk_idx)
            result_list.append({
                "id": result_id,
                "headers": response_content,
            })

    
    sorted_result_list = sorted(result_list, key=lambda x: (x['id'], 0) if isinstance(x['id'], int) else x['id'])

    dst_jsonl = src_jsonl.replace("_result.jsonl", "_parsed.jsonl")
    assert not os.path.exists(dst_jsonl)
    with open(dst_jsonl, 'w') as dst_file:
        dst_file.write('\n'.join([json.dumps(res) for res in sorted_result_list]) + '\n')


if __name__ == "__main__":
    dataset = "finance"
    src_jsonl = os.path.join("/home/ruiying/SHTRAG/data", dataset, "llm_txt", "llm_txt", "sht_result.jsonl")
    parse_batch(src_jsonl)