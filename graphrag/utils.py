import os
import json


def get_document_txt(dataset, doc_filename):
    assert dataset in ["civic", "contract", "qasper", "finance"], dataset
    
    sht_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset,
        "sbert.gpt-4o-mini.c100.s100", 
        "sht", 
        doc_filename + ".json"
    )


    with open(sht_path, 'r') as file:
        sht = json.load(file)

    assert "full_text" in sht
    full_text: str = sht["full_text"]
    return full_text
