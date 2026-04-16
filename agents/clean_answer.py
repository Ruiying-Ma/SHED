import os
import json

if __name__ == "__main__":
    answer_path = "/home/ruiying/SHTRAG/agents/results/core/gpt-5.4/llm_txt_sht/react_agent/contract.jsonl"
    queries = []
    with open(answer_path, 'r') as file:
        for l in file:
            record = json.loads(l)
            if record['is_success'] == False and "ModelCallLimitExceededError" in record['message']:
                continue
            else:
                queries.append(record)
    sorted_queries = sorted(queries, key=lambda x: x['id'])
    # print(len(sorted_queries))
    with open(answer_path, 'w') as file:
        file.write('\n'.join(json.dumps(record) for record in sorted_queries))
    print(len(sorted_queries))