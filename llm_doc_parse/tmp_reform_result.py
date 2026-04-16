import json

item_list = []
pdf_id_set = set()
with open("/home/ruiying/SHTRAG/data/finance/llm_txt/llm_txt/sht_parsed_old.jsonl", 'r') as file:
    for l in file:
        item = json.loads(l)
        item_id = item['id'] if isinstance(item['id'], int) else item['id'][0]
        if item_id not in pdf_id_set:
            pdf_id_set.add(item_id)
        if isinstance(item['id'], int):
            item['id'] = len(pdf_id_set) - 1
        else:
            item['id'][0] = len(pdf_id_set) - 1
        item_list.append(item)

with open("/home/ruiying/SHTRAG/data/finance/llm_txt/llm_txt/sht_parsed.jsonl", 'w') as file:
    for item in item_list:
        file.write(json.dumps(item) + '\n')
    


