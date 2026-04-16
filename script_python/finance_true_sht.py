import os
import json


# clustering_folder = "/home/ruiying/SHTRAG/data/finance/node_clustering"
# for filename in sorted(os.listdir(clustering_folder)):
#     if filename != "VERIZON_2022_10K.json":
#         continue
#     clustering_file_path = os.path.join(clustering_folder, filename)
#     dst_file_path = os.path.join("/home/ruiying/SHTRAG/data/finance/intrinsic", "human_label", filename.replace(".json", ".csv"))
#     os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
#     # assert not os.path.exists(dst_file_path)
#     with open(clustering_file_path, 'r') as file:
#         clustering_info = json.load(file)

#     header_str_list = []
#     last_cluster_id = 0
#     for item in clustering_info:
#         if "Table of Contents" in item['text']:
#             continue
#         if len(item['text']) == 1:
#             continue
#         try:
#             tmp = int(item['text'])
#             continue
#         except:
#             pass
#         if "cluster_id" in item:
#             is_list = (item['type'].strip() == "List item")
#             cluster_id = item['cluster_id']
#             # header_prefix = 'h' if is_list == False else 'l'
#             header_prefix = "h"
#             if item["text"].startswith("ITEM") or item['text'].startswith("Item"):
#                 cluster_id = 2
#             if item["text"].startswith("PART"):
#                 cluster_id = 1
#             header_str = header_prefix + "," + " " * cluster_id + item["text"] + "," + str(item['id'])
#             header_str_list.append(header_str)
#             last_cluster_id = cluster_id
#         elif len(item['text'].strip().split()) <= 10:
#             header_prefix = 'h'
#             header_str = header_prefix + "," + " " * last_cluster_id + item["text"] + "," + str(item['id'])
#             header_str_list.append(header_str)


#     with open(dst_file_path, 'w') as file:
#         file.write("\n".join(header_str_list))

    
# ###################pymupdf get_toc
import pymupdf
import json


def clean_unicode_string(s: str) -> str:
    # Decode all escaped Unicode sequences
    # decoded = s.encode('utf-8').decode('unicode_escape')
    # Replace non-breaking spaces and normalize whitespace
    cleaned = s.replace('\u2013', '-').strip()
    return cleaned
    # return s

filename = "JPMORGAN_2023Q2_10Q"
doc_path = f"/home/ruiying/SHTRAG/data/finance/pdf/{filename}.pdf"
clucstering_path = f"/home/ruiying/SHTRAG/data/finance/node_clustering/{filename}.json"
dst_csv_path = f"/home/ruiying/SHTRAG/data/finance/intrinsic/human_label/{filename}.csv"
document = pymupdf.open(doc_path)
toc = document.get_toc()

with open(clucstering_path, 'r') as file:
    extracted_items = sorted(json.load(file), key=lambda ei: ei['id'])

header_str_list = []
for item in toc:
    level = item[0]
    header = item[1]
    header_id = "?"
    for ei in extracted_items:
        if clean_unicode_string(ei['text'].lower().strip()) == clean_unicode_string((header.lower().strip())):
            header = ei['text']
            header_id = str(ei['id'])
            break
    
    header_str_list.append("h," + " " * (level) + header + "," + header_id)

with open(dst_csv_path, 'w') as file:
    file.write("\n".join(header_str_list))
