import os
import json

def get_doc_txt(dataset, sht_json_filename: str):
    assert dataset in ["civic", "contract", "qasper", "finance"], dataset
    
    sht_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset,
        "sbert.gpt-4o-mini.c100.s100", 
        "sht", 
        sht_json_filename + ".json"
    )


    with open(sht_path, 'r') as file:
        sht = json.load(file)

    assert "full_text" in sht
    full_text: str = sht["full_text"]
    
    return full_text


if __name__ == "__main__":
    for dataset in ["civic", "contract", "qasper"]:
        root_folder = f"/home/ruiying/SHTRAG/data/{dataset}/pdf"
        for filename in sorted(os.listdir(root_folder)):
            assert filename.endswith(".pdf"), filename
            base_filename = filename.replace(".pdf", "")
            full_text = get_doc_txt(dataset, base_filename)
            dst_txt_path = f"/home/ruiying/SHTRAG/graphrag-pypi/{dataset}/input/{base_filename}.txt"
            assert not os.path.exists(dst_txt_path), dst_txt_path
            os.makedirs(os.path.dirname(dst_txt_path), exist_ok=True)
            with open(dst_txt_path, 'w') as f:
                f.write(full_text)
