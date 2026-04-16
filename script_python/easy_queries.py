import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging_config
import numpy as np
from structured_rag import split_text_into_sentences
import eval.utils
from collections import Counter


def _sim_small_chunk(sht_chunk: str, true_prov: str):
    def _norm_str(s: str):
        return eval.utils.remove_articles(eval.utils.remove_punc(eval.utils.lower(s)))

    sht_chunk_tokens = _norm_str(sht_chunk).split()
    true_prov_tokens = _norm_str(true_prov).split()
    common = Counter(sht_chunk_tokens) & Counter(true_prov_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(sht_chunk_tokens)
    recall = 1.0 * num_same / len(true_prov_tokens)
    return max(precision, recall)

def _sim(txt, prov):
    txt_sentences = split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], txt)
    prov_sentences = split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], prov)

    txt_sentences = [s for s in txt_sentences if len(s.strip()) > 0]
    prov_sentences = [s for s in prov_sentences if len(s.strip()) > 0]

    prov_scores = []
    for prov_s in prov_sentences:
        max_score = 0.0
        for txt_s in txt_sentences:
            score = _sim_small_chunk(txt_s, prov_s)
            if score > max_score:
                max_score = score
        prov_scores.append(max_score)
    
    if len(prov_scores) == 0:
        return 0.0
    return np.mean(prov_scores)


def _clean_txt(txt):
    return (" ".join([eval.utils.white_space_fix(s).strip() for s in split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], txt) if len(s.strip()) > 0]))

def _load_nodes(dataset, filename, sht_type):
    node_clustering_path = os.path.join(
        "/home/ruiying/SHTRAG/data",
        dataset,
        sht_type if sht_type != "shed" else "",
        "node_clustering",
        filename + ".json"
    )
    assert os.path.exists(node_clustering_path), node_clustering_path
    logging.info(f"Loading SHT for {filename} ({sht_type}) from {node_clustering_path}...")
    
    with open(node_clustering_path, 'r') as file:
        raw_sht_nodes = json.load(file)

    sht_nodes = []
    for raw_node in raw_sht_nodes:
        if raw_node['type'] in ['Title', 'Section header', 'List item']:
            new_node = {
                'id': len(sht_nodes),
                'type': 'head' if raw_node['type'] != 'List item' else 'list',
                'raw_text': raw_node['text'],
                "page_number": [raw_node['page_number']],
            }
            sht_nodes.append(new_node)
        else:
            if len(sht_nodes) == 0 or sht_nodes[-1]['type'] != 'text':
                new_node = {
                    'id': len(sht_nodes),
                    'type': 'text',
                    'raw_text': raw_node['text'],
                    "page_number": [raw_node['page_number']],
                }
                sht_nodes.append(new_node)
            else:
                sht_nodes[-1]['raw_text'] += "\n" + raw_node['text']
                sht_nodes[-1]['page_number'].append(raw_node['page_number'])
    
    assert [node['id'] for node in sht_nodes] == list(range(len(sht_nodes)))
    assert all([node['type'] in ['head', 'list', 'text'] for node in sht_nodes])
    for node in sht_nodes:
        assert 'clean_text' not in node
        node['clean_text'] = _clean_txt(node['raw_text'])

    last_head_id = -1
    for node in sht_nodes:
        assert 'parent' not in node
        node['parent'] = last_head_id
        if node['type'] != 'text':
            last_head_id = node['id']

    return sht_nodes


def _get_true_prov_list(query_info, dataset):
    if dataset != 'qasper':
        # context = list[str]
        assert isinstance(query_info['context'], list)
        assert all([isinstance(ctx, str) for ctx in query_info['context']])
        prov_list = query_info['context']
    else:
        # context = list[list[str]]
        assert isinstance(query_info['context'], list)
        assert all([isinstance(cl, list) for cl in query_info['context']])
        assert all([isinstance(ctx, str) for cl in query_info['context'] for ctx in cl])
        candid_provs = [x for x in query_info['context'] if len(x) > 0]
        if len(candid_provs) == 0:
            return []
        else:
            prov_list = min(candid_provs, key=lambda x: len(x))
    
    true_prov_list = [_clean_txt(prov) for prov in prov_list]
    result = [t.strip() for t in true_prov_list if len(t.strip()) > 0]
    return result

def _blip_sht_context(all_nodes, true_prov_list, start_list, end_list, dataset, query_id):
    assert len(true_prov_list) == len(start_list) == len(end_list)
    assert [n['id'] for n in all_nodes] == [all_nodes[0]['id'] + i for i in range(len(all_nodes))]
    # sht = true SHT
    # find ctx for each prov, then concat them, sorted by reading order
    THRESH_P = 0.95
    prov_nodes = []
    for prov, start, end in zip(true_prov_list, start_list, end_list):
        assert len(prov.strip()) > 0
        node_ids = []
        m_node_score = dict()
        for node in all_nodes[start:end]:
            node_text = node['clean_text']
            if len(node_text.strip()) == 0:
                continue
            score = _sim(node_text, prov)
            m_node_score[node['id']] = score
            if score >= THRESH_P:
                node_ids.append(node['id'])
        if len(node_ids) > 0:
            # if node_ids == [83, 85]:
            #     node_ids = [83]
            # if node_ids == [37, 39]:
            #     node_ids = [39]
            assert sorted(node_ids) == node_ids
            if node_ids != [node_ids[0] + i for i in range(len(node_ids))]:
                # new_node_ids = []
                # prov_sentences = split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], prov)
                # prov_longest_sentence = max(prov_sentences, key=lambda x: len(x.strip())).strip()
                # for nid in node_ids:
                #     # print(f"Checking node {nid} for longest sentence match: {prov_longest_sentence}")
                #     if prov_longest_sentence in all_nodes[nid]['clean_text']:
                #         # print(f"Found match in node {nid}: {all_nodes[nid]['clean_text']}")
                #         new_node_ids.append(nid)
                # if len(new_node_ids) == 0 or new_node_ids != [new_node_ids[0] + i for i in range(len(new_node_ids))]:
                #     if len(prov_nodes) > 0 and any([((nid in node_ids) or (nid + 1 in node_ids)) for pn in prov_nodes for nid in pn]):
                #         # reuse previous nodes
                #         candidate_ids = set([nid for pn in prov_nodes for nid in pn] + [nid + 1 for pn in prov_nodes for nid in pn])
                #         node_ids = [max([id for id in candidate_ids if id in node_ids])]
                #     else:
                #         if len(prov_nodes) == 0:
                #             node_ids_str = (input(f"Split by commas:\n{prov}\nprov_list: {prov_nodes}\n{node_ids}\n{[all_nodes[n]['clean_text'] for n in node_ids]}"))
                #             node_ids = []
                #             for ns in node_ids_str.split(','):
                #                 ns = ns.strip()
                #                 if len(ns) == 0:
                #                     continue
                #                 node_ids.append(int(ns))
                #         else:
                #             node_ids = []
                # else:
                #     node_ids = new_node_ids
                node_ids_str = (input(f"Split by commas:\n{prov}\nprov_list: {prov_nodes}\n{node_ids}\n{[all_nodes[n]['clean_text'] for n in node_ids]}"))
                node_ids = []
                for ns in node_ids_str.split(','):
                    ns = ns.strip()
                    if len(ns) == 0:
                        continue
                    node_ids.append(int(ns))
            if len(node_ids) == 0 and len(prov_nodes) > 0:
                # reuse previous nodes
                node_ids = [max(prov_nodes[-1])]
            assert node_ids == [node_ids[0] + i for i in range(len(node_ids))], f"{prov}\n{node_ids}\n{[all_nodes[n]['clean_text'] for n in node_ids]}"
        if len(node_ids) == 0:
            # return False
            sorted_node_ids = sorted(list(m_node_score.keys()), key=lambda x: m_node_score[x], reverse=True)
            if len(sorted_node_ids) == 0:
                logging.warning(f"No matching nodes found for prov: {prov}")
                return False
            best_node_id = sorted_node_ids[0]
            node_ids = [best_node_id]
        prov_nodes.append(node_ids)
    
    
    assert len(prov_nodes) == len(true_prov_list)
    

    flattened_node_ids = set([n for nl in prov_nodes for n in nl])
    # easy or not
    header_node_ids = [n for n in flattened_node_ids if all_nodes[n]['type'] != 'text']
    txt_node_ids = [n for n in flattened_node_ids if all_nodes[n]['type'] == 'text']
    txt_node_parent_ids = [all_nodes[n]['parent'] for n in txt_node_ids if all_nodes[n]['parent'] != -1]
    header_ids_not_used_by_txt = [hid for hid in header_node_ids if hid not in txt_node_parent_ids]
    if len(header_ids_not_used_by_txt) > 0:
        is_easy = True
        for hid in header_ids_not_used_by_txt:
            if (hid + 1) not in flattened_node_ids and (hid - 1) not in flattened_node_ids:
                is_easy = False
                break
    else:
        is_easy = True
    
    if len(true_prov_list) == 0:
        is_easy = False
    if len(flattened_node_ids) == 1:
        is_easy = True

    if len(flattened_node_ids) > 0 and sorted(flattened_node_ids) == list(range(min(flattened_node_ids), max(flattened_node_ids) + 1)):
        assert is_easy == True

    dst_path = f"easy_queries_{dataset}.jsonl"
    with open(dst_path, 'a') as file:
        json_line = {
            "id": query_id,
            "is_easy": is_easy,
            "prov": [
                {
                    "true_prov": true_prov_list[i],
                    "matched_nodes": [
                        {
                            "id": nid,
                            "type": all_nodes[nid]['type'],
                            "parent": all_nodes[nid]['parent'],
                            "clean_text": all_nodes[nid]['clean_text']
                        }
                        for nid in prov_nodes[i]
                    ],
                } for i in range(len(true_prov_list))
            ]
        }
        file.write(json.dumps(json_line) + "\n")

    return is_easy

    

def blip(dataset):
    query_path = os.path.join(
        "/home/ruiying/SHTRAG/data",
        dataset,
        "queries.json" if dataset != 'finance' else "queries_pages.json",
    )


    # blip_path = query_path.replace("queries.json", "blip.jsonl")
    # existing_query_ids = set()
    # if os.path.exists(blip_path):
    #     with open(blip_path, 'r') as file:
    #         for line in file:
    #             json_line = json.loads(line)
    #             existing_query_ids.add(json_line['id'])

    with open(query_path, 'r') as file:
        queries_info = json.load(file)

    hard_queries = []
    for qid, query_info in enumerate(queries_info):
        # if qid in existing_query_ids:
        #     print(f"Skipping {qid} (already processed)...")
        #     continue
        file_name = query_info["file_name"]
        print(f"Processing {qid}...")
        sht_type = "intrinsic" if dataset != 'civic' else 'wide'
        sht = _load_nodes(dataset, file_name, sht_type)
        true_prov_list = _get_true_prov_list(query_info, dataset)
        if dataset == 'finance':
            query_pages = query_info['context_pages']
            assert len(query_pages) == len(true_prov_list)
            start_list = []
            end_list = []
            for query_page in query_pages:
                candid_node_ids = []
                for node in sht:
                    node_pages = node['page_number']
                    if query_page in node_pages:
                        candid_node_ids.append(node['id'])
                assert len(candid_node_ids) > 0
                start = candid_node_ids[0]
                end = candid_node_ids[-1] + 1
                assert candid_node_ids == list(range(start, end)), f"{query_pages}\n{candid_node_ids}\n{[sht[i]['page_number'] for i in range(start, end)]}"
                start_list.append(start)
                end_list.append(end)
        else:
            start_list = [0 for _ in true_prov_list]
            end_list = [len(sht) for _ in true_prov_list]
        is_easy = _blip_sht_context(sht, true_prov_list, start_list, end_list, dataset, qid)
        print(f"Query {qid} is {'easy' if is_easy else 'hard'}.")
        if is_easy == False:
            hard_queries.append(qid)
    
    print(f"Hard queries: {hard_queries}")

                

if __name__ == "__main__":
    logging.disable(logging.INFO)
    blip("civic")
    