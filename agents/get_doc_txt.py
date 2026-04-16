import pymupdf, pymupdf4llm
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import tiktoken

from config import DATA_ROOT_FOLDER, DATASET_LIST, get_cost_usd

def get_doc_txt(dataset, filename):
    sht_path = Path(DATA_ROOT_FOLDER) / dataset / "sbert.gpt-4o-mini.c100.s100" / "sht" / (filename + ".json")

    with open(sht_path, 'r') as file:
        sht = json.load(file)

    assert "full_text" in sht
    full_text: str = sht["full_text"]

    return full_text

if __name__ == "__main__":
    model = "gpt-5.4"

    encoder = tiktoken.get_encoding("cl100k_base")

    cost_details = dict()

    for dataset in DATASET_LIST:
        cost_details[dataset] = {
            "total_cost": 0,
        }

        dataset_folder = Path(DATA_ROOT_FOLDER) / dataset
        query_path = dataset_folder / "queries.json"

        with open(query_path, 'r') as file:
            queries = json.load(file)

        for q in queries:
            file_name = q["file_name"]
            doc_txt = get_doc_txt(dataset, file_name)
            answer = q["answer"]
            input_tokens = len(encoder.encode(doc_txt))
            try:
                output_tokens = len(encoder.encode(answer)) * 5
            except Exception as e:
                output_tokens = 20
            cost = get_cost_usd(model, input_tokens, 0, output_tokens)
            cost_details[dataset][q['id']] = {
                "file_name": file_name,
                "input_tokens": input_tokens,
                "cost": cost
            }
            cost_details[dataset]["total_cost"] += cost
    

    with open("estimated_cost.json", 'w') as f:
        json.dump(cost_details, f, indent=2)

            
