import json
from rank_bm25 import BM25Okapi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structured_rag import split_text_into_sentences
import tiktoken
from copy import deepcopy
from pathlib import Path

TOKENIZER = tiktoken.get_encoding("cl100k_base")

DATASET = None


def _get_doc_txt(dataset, file_name: str) -> str:
    DATA_ROOT_FOLDER = "/home/ruiying/SHTRAG/data"
    with open(Path(DATA_ROOT_FOLDER) / dataset / "intrinsic" / "toc_textspan_clean" / (file_name + ".json"), 'r') as file:
            full_text = json.load(file)['1'].strip()
            return full_text


def _get_level(header_str: str):
    assert header_str.startswith("#")
    level = -1
    for cid, char in enumerate(header_str):
        if char != "#":
            assert level >= 0
            return level, header_str[cid:].strip()
        level += 1
    
    assert level >= 0
    return level, ""

def _parse_md(raw_md_str_list: list):
    md_lines = []
    for raw_md_str in raw_md_str_list:
        if raw_md_str.strip().startswith("```markdown") and raw_md_str.strip().endswith("```"):
            md_str = raw_md_str.strip()[len("```markdown"): -len("```")]
        else:
            md_str = raw_md_str
        md_lines += md_str.splitlines()
    
    header_list = []
    cur_line = ""
    for line in md_lines:
        if line.startswith("#"):
            if cur_line.startswith("#"):
                header_list.append(cur_line)
            cur_line = line
        else:
            cur_line += line
    if cur_line.startswith("#"):
        header_list.append(cur_line)


    header_info_list = []
    for header_str in header_list:
        level, header = _get_level(header_str)
        
        header_info = {
            "id": len(header_info_list),
            "level": level,
            "header": header,
            "raw_header": header_str,
        }

        for hi in header_info_list[::-1]:
            assert hi['id'] < header_info['id']
            if hi['level'] < header_info['level']:
                assert 'parent_id' not in header_info
                header_info['parent_id'] = hi['id']
                break
        if 'parent_id' not in header_info:
            header_info['parent_id'] = -1

        header_info_list.append(header_info)
    
    assert len(header_info_list) == len(header_list)
    assert [hi['id'] for hi in header_info_list] == list(range(len(header_list)))


    m_id_cluster = dict()
    for hi in header_info_list:
        parent_id = hi['parent_id']
        if parent_id not in m_id_cluster:
            new_cid = len(m_id_cluster)
            assert new_cid not in list(m_id_cluster.values())
            m_id_cluster[parent_id] = new_cid

    for hi in header_info_list:
        parent_id = hi['parent_id']
        assert parent_id in m_id_cluster
        assert 'cluster_id' not in hi
        hi['cluster_id'] = m_id_cluster[parent_id]

    return header_info_list

def _find_all_occurrences(text: str, substring: str) -> list[int]:
    indices = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index == -1:
            break
        indices.append(index)
        start = index + 1  # move one char forward to allow overlapping matches
    
    if len(indices) > 0:
        assert sorted(indices) == indices
        num = len(indices)
        if DATASET == "finance":
            num = min(3, num)
        return indices[:num], [deepcopy(substring) for _ in indices[:num]]
    
    raw_sentences = split_text_into_sentences([".", "!", "?", "\n"], text)
    assert all(len(s) > 0 for s in raw_sentences), raw_sentences
    sentence_indices = []
    sentences = []
    cur_idx = 0
    for sent in raw_sentences:
        if sent in text[cur_idx:]:
            idx = text.find(sent, cur_idx)
            assert idx >= cur_idx, f"idx={idx}, cur_idx={cur_idx}, sent={sent}, last_sent={sentences[-1] if len(sentences) > 0 else 'N/A'}"
            sentence_indices.append(idx)
            sentences.append(sent)
            assert len(sent) > 0
            cur_idx = idx + len(sent)
    assert len(raw_sentences) == len(sentence_indices), f"{len(raw_sentences)} vs {len(sentence_indices)}"
    
    token_sent = [TOKENIZER.encode(s) for s in sentences]
    token_substring = TOKENIZER.encode(substring)
    scores = BM25Okapi(token_sent).get_scores(token_substring)
    sorted_sent_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    candid_sent_indices = [i for i in sorted_sent_indices if scores[i] > 0.8 and len(sentences[i].strip()) > 0]
    candid_indices = [sentence_indices[i] for i in candid_sent_indices]
    if DATASET == "finance":
        candid_len = min(2, len(candid_indices))
    else:
        candid_len = min(5, len(candid_indices))
    return candid_indices[:candid_len], [deepcopy(sentences[i]) for i in candid_sent_indices[:candid_len]]
            

def _get_node_clustering(raw_md_str_list, file_name, dataset):
    assert isinstance(raw_md_str_list, list)
    print(f"Clustering {file_name}...")
    doc_txt = _get_doc_txt(dataset, file_name)

    header_info_list = _parse_md(raw_md_str_list)
    assert [h['id'] for h in header_info_list] == list(range(len(header_info_list)))

    print(f"\tParsed header: {len(header_info_list)}")

    header_pos_list = []
    candid_header_list = []
    for hi in header_info_list:
        header = hi['header']
        header_pos, candid_header = _find_all_occurrences(doc_txt, header)
        assert len(header_pos) == len(candid_header)
        header_pos_list.append(header_pos)
        candid_header_list.append(candid_header)


    assert len(header_info_list) == len(header_pos_list) == len(candid_header_list)

    dp_info_dict = dict()

    for header_info, header_pos, candid_headers in zip(header_info_list, header_pos_list, candid_header_list):
        hid = header_info['id']
        assert hid not in dp_info_dict
        dp_info_dict[hid] = dict()
        for p_idx, pid in enumerate(header_pos):
            assert pid not in dp_info_dict[hid], f"{header_pos}, {candid_headers}"
            dp_info_dict[hid][pid] = {
                "len": 1,
                "prev_hid": None,
                "prev_pid": None,
            }
            for other_hid in dp_info_dict:
                if other_hid == hid:
                    continue
                assert other_hid < hid
                for other_p_idx, other_pid in enumerate(dp_info_dict[other_hid]):
                    if other_pid + len(candid_header_list[other_hid][other_p_idx]) > pid:
                        continue
                    cur_len = dp_info_dict[other_hid][other_pid]["len"] + 1
                    if cur_len > dp_info_dict[hid][pid]["len"]:
                        dp_info_dict[hid][pid]["len"] = cur_len
                        dp_info_dict[hid][pid]["prev_hid"] = other_hid
                        dp_info_dict[hid][pid]["prev_pid"] = other_pid
    
    # backtrack to find the longest path
    max_len = -1
    max_hid = None
    max_pid = None
    for hid in dp_info_dict:
        for pid in dp_info_dict[hid]:
            if dp_info_dict[hid][pid]["len"] > max_len:
                max_len = dp_info_dict[hid][pid]["len"]
                max_hid = hid
                max_pid = pid

    print(f"\tLongest path length: {max_len}")

    if max_len == -1:
        return [
            {
                "id": 0,
                "text": doc_txt.strip(),
                "type": "Text",
                "features": {}
            }
        ]

    assert max_len > 0
    assert max_hid != None
    assert max_pid != None

    longest_path = []
    cur_hid = max_hid
    cur_pid = max_pid
    while cur_hid != None and cur_pid != None:
        longest_path.append((cur_hid, cur_pid))
        next_hid = dp_info_dict[cur_hid][cur_pid]["prev_hid"]
        next_pid = dp_info_dict[cur_hid][cur_pid]["prev_pid"]
        cur_hid = next_hid
        cur_pid = next_pid

    assert len(longest_path) == max_len
    longest_path = longest_path[::-1]  # reverse to get the correct order
    
    pos_list = [pid for hid, pid in longest_path]
    assert sorted(pos_list) == pos_list
    
    
    cluster_list = []

    prev_idx = 0
    for hid, pid in longest_path:
        assert prev_idx <= pid
        prev_text = doc_txt[prev_idx:pid]
        cand_header = candid_header_list[hid][header_pos_list[hid].index(pid)]
        assert doc_txt[pid:].startswith(cand_header)
        txt_cluster = {
            "id": len(cluster_list),
            "text": prev_text.strip(),
            "type": "Text",
            "features": {}
        }
        if len(prev_text.strip()) > 0:
            cluster_list.append(txt_cluster)
        header_cluster = {
            "id": len(cluster_list),
            "text": cand_header.strip(),
            "type": "Section header",
            "features": {},
            "cluster_id": header_info_list[hid]['cluster_id'],
        }
        if len(header_cluster['text'].strip()) > 0:
            cluster_list.append(header_cluster)
        prev_idx = pid + len(cand_header)

    assert [c['id'] for c in cluster_list] == list(range(len(cluster_list)))

    return cluster_list, max_len / len(header_info_list)

def get_node_clustering(dataset):
    root_folder = os.path.join("/home/ruiying/SHTRAG/data", dataset)
    llm_vision_folder = os.path.join(root_folder, "llm_vision_sht") 
    os.makedirs(llm_vision_folder, exist_ok=True)
    with open(os.path.join("/home/ruiying/SHTRAG/agents/results/llm_gen_toc_response/gpt-5.4/llm_vision", f"{dataset}.jsonl"), 'r') as file:
        raw_md_list = [json.loads(line) for line in file]
    
    m_file_toc = dict()
    for raw_md_info in raw_md_list:
        if raw_md_info['is_success'] == False:
            continue
        file_name = raw_md_info['file_name']
        if file_name not in m_file_toc:
            m_file_toc[file_name] = []
        m_file_toc[file_name].append(raw_md_info['message'])
    
    parse_rate_list = []

    for file_name, raw_md_list in m_file_toc.items():
        dst_path = os.path.join(llm_vision_folder, "node_clustering", f"{file_name}.json")
        if os.path.exists(dst_path):
            print(f"Node clustering already exists for {file_name}, skipping...")
            continue
        cluster_list, parse_rate = _get_node_clustering(raw_md_list, file_name, dataset)
        parse_rate_list.append(parse_rate)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # assert not os.path.exists(dst_path)
        
        with open(dst_path, 'w') as file:
            json.dump(cluster_list, file, indent=4)

    print(parse_rate_list)
    try:
        print(f"Average parse rate: {sum(parse_rate_list) / len(parse_rate_list)}")
    except ZeroDivisionError:
        print("No successful parsing.")


    # root_folder = os.path.join("/home/ruiying/SHTRAG/data", dataset)
    # pdf_filename_list = sorted(os.listdir(os.path.join(root_folder, "pdf")))
    # llm_txt_folder = os.path.join(root_folder, "llm_vision") 
    # raw_md_list = []
    # for pdf_filename in pdf_filename_list:
    #     jsonl_path = os.path.join(llm_txt_folder, "llm_vision", pdf_filename.replace(".pdf", ".jsonl"))
    #     assert os.path.exists(jsonl_path), jsonl_path
    #     cur_raw_md_info_list = []

    #     with open(jsonl_path, 'r') as file:
    #         for line in file:
    #             cur_raw_md_info = json.loads(line)
    #             cur_raw_md_info_list.append(cur_raw_md_info)
        
    #     sorted_cur_raw_md_info_list = sorted(cur_raw_md_info_list, key=lambda x: int(x['id']))
    #     raw_md_list.append({
    #         "id": len(raw_md_list),
    #         "headers": [r['headers'] for r in sorted_cur_raw_md_info_list],
    #     })
        
    # # with open(os.path.join(llm_txt_folder, "llm_vision", "sht_parsed.jsonl"), 'r') as file:
    # #     raw_raw_md_list = [json.loads(line) for line in file]

    # # raw_md_list = []
    # # pdf_id_set = set()
    # # for raw_raw_md_info in raw_raw_md_list:
    # #     pdf_id = raw_raw_md_info['id'] if isinstance(raw_raw_md_info['id'], int) else raw_raw_md_info['id'][0]
    # #     assert [r['id'] for r in raw_md_list] == list(range(len(raw_md_list))) == list(sorted(pdf_id_set))
    # #     if pdf_id not in pdf_id_set:
    # #         pdf_id_set.add(pdf_id)
    # #         assert pdf_id == (raw_md_list[-1]['id'] + 1 if len(raw_md_list) > 0 else 0), f"{pdf_id} vs {raw_md_list[-1]['id'] if len(raw_md_list) > 0 else 'N/A'}"
    # #         raw_md_list.append({
    # #             "id": pdf_id,
    # #             "headers": [raw_raw_md_info['headers']],
    # #         })
    # #     else:
    # #         last_md = raw_md_list[-1]
    # #         assert pdf_id == last_md['id'], f"{pdf_id} vs {last_md['id']}"
    # #         last_md['headers'].append(raw_raw_md_info['headers'])
    
    
    # assert len(raw_md_list) == len(pdf_filename_list)
    # assert [r['id'] for r in raw_md_list] == list(range(len(raw_md_list))) 

    # parse_rate_list = []

    # for raw_md_info, pdf_filename in zip(raw_md_list, pdf_filename_list):
    #     pdf_path = os.path.join(root_folder, "pdf", pdf_filename)
    #     dst_path = os.path.join(llm_txt_folder, "node_clustering", pdf_filename.replace(".pdf", ".json"))
    #     if os.path.exists(dst_path):
    #         print(f"Node clustering for {pdf_filename} is already existed.")
    #         continue
    #     cluster_list, parse_rate = _get_node_clustering(raw_md_info['headers'], pdf_path)
    #     parse_rate_list.append(parse_rate)
    #     os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #     assert not os.path.exists(dst_path)
    #     with open(dst_path, 'w') as file:
    #         json.dump(cluster_list, file, indent=4)

    # print(parse_rate_list)
    # print(f"Average parse rate: {sum(parse_rate_list) / len(parse_rate_list)}")

if __name__ == "__main__":
    for dataset in [
        # "civic_rand_v1",
        # "contract_rand_v0_1",
        "finance_rand_v1",
        "qasper_rand_v1",

    ]:
        get_node_clustering(dataset)