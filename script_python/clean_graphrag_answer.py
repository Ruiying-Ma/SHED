import os
import json


if __name__ == "__main__":
    m_dataset_answer = {
        "civic": 418,
        "contract": 1241,
        "qasper": 1451,
        "finance": 150,
    }

    dataset = "finance"

    answer_orig_path = f"/home/ruiying/SHTRAG/graphrag/{dataset}/answer_orig.jsonl"

    dst_path = f"/home/ruiying/SHTRAG/data/{dataset}/baselines/graphrag/answer.jsonl"
    assert not os.path.exists(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(answer_orig_path, 'r') as file:
        raw_answer_list = [json.loads(l) for l in file.readlines()]

    m_id_raw_answer = {
        raw_answer["id"]: raw_answer
        for raw_answer in raw_answer_list
    }

    sorted_answer_list = []
    for id in range(m_dataset_answer[dataset]):
        if id in m_id_raw_answer:
            sorted_answer_list.append(m_id_raw_answer[id])
        else:
            sorted_answer_list.append({
                "id": id,
                "answer": None
            })
    assert list(a["id"] for a in sorted_answer_list) == list(range(m_dataset_answer[dataset]))

    

    with open(dst_path, 'w') as file:
        file.write("\n".join([json.dumps(a) for a in sorted_answer_list]) + "\n")
    