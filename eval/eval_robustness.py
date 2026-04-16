import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
from structured_rag import utils
import eval.utils as eval_utils
from collections import Counter
import numpy as np
import logging
import logging_config

DELIMETER = " "
CHUNK_SIZE = 20


def _normalize_str(s: str):
    return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))

def _compare_str(str1: str, str2: str) -> float:
    '''
    str2 = true
    '''
    str1_tokens = _normalize_str(str1).split()
    str2_tokens = _normalize_str(str2).split()
    common = Counter(str1_tokens) & Counter(str2_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
    precision = num_same / len(str1_tokens)
    recall = num_same / len(str2_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

def _clean_node(node_dict):
    new_node = {
        'id': node_dict['id'],
        'is_dummy': node_dict['is_dummy'],
        'nondummy_parent': node_dict['nondummy_parent']
    }
    if node_dict['is_dummy'] == False:
        new_node['type'] = node_dict['type']
        if node_dict['type'] == 'text':
            new_node['texts'] = node_dict['texts']
        else:
            assert node_dict['type'] in ['head', 'list']
            new_node['texts'] = [node_dict['heading']]
    
    return new_node

# true, grobid, shed, llm
def _load_sht(dataset, filename, sht_type):
    sht_path = os.path.join(
        config.DATA_ROOT_FOLDER,
        dataset,
        sht_type if sht_type != "shed" else "",
        f"sbert.gpt-4o-mini.c100.s100",
        "sht" if sht_type not in ['wide', 'deep', 'llm_txt'] else "sht_skeleton",
        filename + ".json"
    )
    assert os.path.exists(sht_path), sht_path
    logging.info(f"Loading SHT for {filename} ({sht_type}) from {sht_path}...")
    
    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    cleaned_sht_nodes = [_clean_node(n) for n in raw_sht_nodes]
    return cleaned_sht_nodes

def _retrieve_nodes(sht, txt, direction):
    retrieved_node_list = []
    for node in sht:
        if node['is_dummy'] == True:
            continue
        # if direction == 'top-down' and node['type'] == 'text':
        #     continue
        # if direction == 'bottom-up' and node['type'] != 'text':
        #     continue
        if txt in "".join(node['texts']):
            retrieved_node_list.append(node['id'])
            break

    return retrieved_node_list

def _get_node_ctx(sht, node_id, direction):
    assert sht[node_id]['is_dummy'] == False
    assert direction in ['bottom-up', 'top-down']

    if direction == 'bottom-up':
        root_to_parent_ids = utils.get_nondummy_ancestors(sht, node_id)
        assert node_id not in root_to_parent_ids
        if sht[node_id]['type'] != 'text':
            root_to_parent_ids.append(node_id)
        assert sorted(root_to_parent_ids) == root_to_parent_ids
        assert all([sht[nid]['type'] != 'text' for nid in root_to_parent_ids])
        assert all([len(sht[nid]['texts']) == 1 for nid in root_to_parent_ids])
        return DELIMETER.join([sht[nid]['texts'][0] for nid in root_to_parent_ids])

    assert direction == 'top-down'
    # retrieve the raw text span
    descendant_ids = [node_id]
    flag = False
    for node in sht:
        if node['is_dummy'] == True:
            continue
        if node['id'] <= node_id:
            continue
        if node_id in utils.get_nondummy_ancestors(sht, node['id']):
            assert flag == False
            descendant_ids.append(node['id'])
        else:
            if flag == False:
                flag = True
            assert node_id not in utils.get_nondummy_ancestors(sht, node['id'])
    assert sorted(descendant_ids) == descendant_ids
    ctx = ""
    for nid in descendant_ids:
        assert isinstance(sht[nid]['texts'], list)
        for t in sht[nid]['texts']:
            assert isinstance(t, str)
            ctx += t
    return ctx

def _load_data(m_type_sht, direction):
    assert direction in ['top-down', 'bottom-up']
    assert 'intrinsic' in m_type_sht
    
    data_list = []
    for node in m_type_sht['intrinsic']:
        if node['is_dummy'] == True:
            continue
        if direction == 'top-down' and node['type'] == 'text':
            continue
        # if direction == "bottom-up" and node['type'] != 'text':
        #     continue
        for cid, chunk in enumerate(node['texts']):
            data_info = {
                'id': len(data_list),
                'txt': chunk,
                'pos': (node['id'], cid) # position in true sht
            }
            for sht_type in m_type_sht:
                if sht_type == 'intrinsic':
                    continue
                assert sht_type not in data_info
                data_info[sht_type] = _retrieve_nodes(m_type_sht[sht_type], chunk, direction)
            if all([len(data_info[sht_type]) > 0 for sht_type in m_type_sht if sht_type != 'intrinsic']):
                data_list.append(data_info)
    assert [di['id'] for di in data_list] == list(range(len(data_list)))
    return data_list

def _comp_score(s1, s2):
    if s1['f1'] > s2['f1']:
        return True
    if s1['f1'] == s2['f1'] and s1['recall'] > s2['recall']:
        return True
    return False

def _agg_score(data_pos_list, data_score_list):
    # select the opt score for each data
    assert len(data_pos_list) == len(data_score_list)
    opt_data_score_list = []
    for ds in data_score_list:
        max_f1_score = None
        for s in ds:
            if max_f1_score == None or _comp_score(s, max_f1_score) == True:
                max_f1_score = s
        opt_data_score_list.append(s)
    assert len(data_pos_list) == len(opt_data_score_list)
    # select the opt data for each node
    m_node_score = dict()
    for data_pos, data_score in zip(data_pos_list, opt_data_score_list):
        node_id = data_pos[0]
        if node_id not in m_node_score:
            m_node_score[node_id] = data_score
        elif _comp_score(data_score, m_node_score[node_id]) == True:
            m_node_score[node_id] = data_score

    # aggregrate
    agg_score = dict()
    for score_name in ['recall', 'precision', 'f1']:
        agg_score[score_name] = np.mean([s[score_name] for s in list(m_node_score.values())])


    return agg_score

    
def _eval_robustness_per_sht(m_type_sht, direction):
    assert direction in ['top-down', 'bottom-up']

    data_list = _load_data(m_type_sht, direction)
    print(f"\t#data = {len(data_list)}")
    if len(data_list) == 0:
        return None, data_list

    for data_info in data_list:
        logging.debug(f"{data_info}")
        assert 'score' not in data_info
        data_info['score'] = dict()
        true_ctx = _get_node_ctx(m_type_sht['intrinsic'], data_info['pos'][0], direction)
        for sht_type in m_type_sht:
            if sht_type == 'intrinsic':
                continue
            retrieved_ctx_list = [
                _get_node_ctx(m_type_sht[sht_type], nid, direction)
                for nid in data_info[sht_type]
            ]
            assert sht_type not in data_info['score']
            data_info['score'][sht_type] = [
                _compare_str(ctx, true_ctx)
                for ctx in retrieved_ctx_list
            ]

    agg_score = dict()
    for sht_type in m_type_sht:
        if sht_type == 'intrinsic':
            continue
        assert sht_type not in agg_score
        data_pos_list = [di['pos'] for di in data_list]
        data_score_list = [di['score'][sht_type] for di in data_list]
        agg_score[sht_type] = _agg_score(data_pos_list, data_score_list)

    return agg_score, data_list


def eval_robustness(dataset, sht_type_list, direction):
    assert 'intrinsic' in sht_type_list

    m_filename_score = dict()

    for pdf_name in sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf'))):
        filename = pdf_name.replace(".pdf", "")
        print(filename)
        m_type_sht = {
            sht_type: _load_sht(dataset, filename, sht_type)
            for sht_type in sht_type_list
        }
        assert filename not in m_filename_score
        score, data_list = _eval_robustness_per_sht(m_type_sht, direction)
        if score != None:
            m_filename_score[filename] = score
            print(f"\t{m_filename_score[filename]}")

    print(f"#files = {len(m_filename_score)}")
    
    agg_score = dict()
    for sht_type in sht_type_list:
        if sht_type == "intrinsic":
            continue
        agg_score[sht_type] = dict()
        for score_name in ['recall', 'precision', 'f1']:
            score_list = [s[sht_type][score_name] for s in m_filename_score.values()]
            agg_score[sht_type][score_name] = round(np.mean(score_list) * 100, 2)
    
    return agg_score



if __name__ == "__main__":
    logging.disable(logging.INFO)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", required=True, type=str, choices=['civic', 'contract', 'qasper', 'finance'])

    method_list = ['intrinsic', 'llm_txt', 'grobid', 'wide', 'deep', 'shed']
    dataset_list = ['civic', 'contract', 'qasper', 'finance']

    for mode in ['bottom-up', 'top-down']:
        print(f"================================{mode}=========================================")
        m_dataset_score = dict()
        for dataset in dataset_list:
            print(f"-----------------------------{dataset}--------------------------")
            score = eval_robustness(dataset, method_list, mode)
            print(score)
            m_dataset_score[dataset] = score
        
        tab_line = ""
        for method in method_list:
            if method == "intrinsic":
                continue
            tab_line += f"{method} & "
            for score_name in ["recall", "precision", "f1"]:
                for dataset in dataset_list:
                    tab_line += f"{m_dataset_score[dataset][method][score_name]}\% & "
            
            tab_line += "\\\\\n"
        
        print("**************TABLE********************")
        print(tab_line)





