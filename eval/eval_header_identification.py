import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
from structured_rag import utils, split_text_into_sentences
import eval.utils as eval_utils
from collections import Counter
import numpy as np
import logging
import logging_config

def _normalize_str(txt: str):
    # return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))
    return (" ".join([eval_utils.white_space_fix(s).strip() for s in split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], txt) if len(s.strip()) > 0]))

# true, grobid, shed, llm
def _load_headers(dataset, filename, sht_type):
    sht_path = os.path.join(
        config.DATA_ROOT_FOLDER,
        dataset,
        sht_type if sht_type != "shed" else "",
        f"sbert.gpt-4o-mini.c100.s100",
        "sht" if sht_type not in ['wide', 'deep', 'llm_txt', 'llm_vision'] else "sht_skeleton",
        filename + ".json"
    )
    assert os.path.exists(sht_path), sht_path
    logging.info(f"Loading SHT for {filename} ({sht_type}) from {sht_path}...")
    
    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    header_list = [
        _normalize_str(node['heading'])
        for node in raw_sht_nodes
        if node['is_dummy'] == False and node['type'] == 'head'
    ]

    return header_list


def _eval_header_identification_per_sht(header_list, true_header_list):
    if len(header_list) == 0 and len(true_header_list) == 0:
        return 1, 1, 1 # recall, precision, f1
    elif len(header_list) == 0 or len(true_header_list) == 0:
        return 0, 0, 0
    
    assert len(header_list) > 0 and len(true_header_list) > 0
    dp_mat = dict()

    for hlen in range(1, len(header_list) + 1):
        for true_hlen in range(1, len(true_header_list) + 1):
            assert (hlen, true_hlen) not in dp_mat
            dp_mat[(hlen, true_hlen)] = dict()
            # not matching header to any true_header
            if hlen > 1:
                dp_mat[((hlen, true_hlen))] = {k: v for k, v in dp_mat[(hlen-1, true_hlen)].items()}
            
            # matching header to some true_header
            h_idx = hlen - 1
            for true_h_idx in range(true_hlen)[::-1]:
                if (header_list[h_idx] in true_header_list[true_h_idx]) or (true_header_list[true_h_idx] in header_list[h_idx]):
                    prev_dict = dp_mat[(hlen - 1, true_h_idx)] if (hlen - 1 > 0 and true_h_idx > 0) else dict()
                    assert h_idx not in prev_dict
                    new_dict = {k: v for k, v in prev_dict.items()}
                    new_dict[h_idx] = true_h_idx
                    if len(new_dict) > len(dp_mat[(hlen, true_hlen)]):
                        dp_mat[(hlen, true_hlen)] = new_dict
                    break
            
    final_match = dp_mat[(len(header_list), len(true_header_list))]
    
    # for k, v in final_match.items():
    #     assert header_list[k] in true_header_list[v], (header_list[k], true_header_list[v])
    

    assert len(set(final_match.keys())) == len(final_match)
    assert len(set(final_match.values())) == len(final_match)
    assert set(final_match.keys()).issubset(set(range(len(header_list))))
    assert set(final_match.values()).issubset(set(range(len(true_header_list))))

    recall = len(final_match) / len(true_header_list)
    precision = len(final_match) / len(header_list)
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return recall, precision, f1


def eval_header_identification(dataset, sht_type):
    recall_list = []
    precision_list = []
    f1_list = []

    for pdf_name in sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf'))):
        filename = pdf_name.replace(".pdf", "")
        true_header_list = _load_headers(dataset, filename, "intrinsic")
        header_list = _load_headers(dataset, filename, sht_type)
        recall, precision, f1 = _eval_header_identification_per_sht(header_list, true_header_list)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        print(f"[{sht_type}][{filename}] Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, #True Headers: {len(true_header_list)}, #Predicted Headers: {len(header_list)}")
    
    return round(np.mean(recall_list), 4), round(np.mean(precision_list), 4), round(np.mean(f1_list), 4)

if __name__ == "__main__":
    dataset_list = ['civic', 'contract', 'qasper', 'finance']
    sht_type_list = ['shed', 'llm_txt', 'llm_vision', 'grobid']

    tab_str = ""
    for sht_type in sht_type_list:
        tab_str += sht_type
        m_dataset_score = dict()
        for dataset in dataset_list:
            recall, precision, f1 = eval_header_identification(dataset, sht_type)
            m_dataset_score[dataset] = (recall, precision, f1)
        for score_id in range(3):
            tab_str += " & " + " & ".join([str(round(100 * (m_dataset_score[dataset][score_id]), 2)) + "\%" for dataset in dataset_list])
        tab_str += " \\\\\n"
    
    print(tab_str)


        
                