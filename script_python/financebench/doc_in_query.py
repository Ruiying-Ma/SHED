import json
import os

def get_doc_in_query():
    doc_folder = "/home/ruiying/financebench/pdfs"
    doc_in_query = []
    with open("/home/ruiying/financebench/data/financebench_open_source.jsonl", 'r') as file:
        for l in file:
            query = json.loads(l)
            doc_name = query["doc_name"] + ".pdf"
            doc_path = os.path.join(doc_folder, doc_name)
            if doc_path in doc_in_query:
                continue
            if os.path.exists(doc_path):
                assert query["dataset_subset_label"] == "OPEN_SOURCE"
                doc_in_query.append(doc_path)
    return doc_in_query

if __name__ == "__main__":
    docs = get_doc_in_query()
    # print(docs)
    print(len(docs))
    