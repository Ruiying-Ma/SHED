import os
import json


raw_qinfo_list = []
qinfo_list = []
with open("/home/ruiying/financebench/data/financebench_open_source.jsonl", 'r') as file:
    for l in file:
        raw_qinfo = json.loads(l)
        raw_qinfo_list.append(raw_qinfo)
    
for raw_qinfo in raw_qinfo_list:
    qinfo = {
        "id": len(qinfo_list),
        "file_name": raw_qinfo["doc_name"],
        "query": raw_qinfo["question"],
        "prompt_template": "Please answer the question below using the provided context.\n\n[Begin of Question]\n" +  raw_qinfo['question'].strip() + '''\n[End of Question]\n\n[Begin of Context]\n{context}\n[End of Context]\n\nYour answer must follow these instructions:\n[Begin of instructions]\n- If the question is yes/no, output only "yes" or "no".\n- If the question asks for a number, output only the number.\n- Otherwise, output exactly one sentence.\n[End of Instructions]\n\nYour answer:''',
        "answer": [raw_qinfo["answer"]],
        "context": [e["evidence_text"] for e in raw_qinfo["evidence"]],
        "context_pages": [e["evidence_page_num"] + 1 for e in raw_qinfo["evidence"]]
    }

    qinfo_list.append(qinfo)

with open("/home/ruiying/SHTRAG/data/finance/queries_pages.json", 'w') as file:
    json.dump(qinfo_list, file, indent=4)

