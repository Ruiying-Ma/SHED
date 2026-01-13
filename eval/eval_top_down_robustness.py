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
from multiprocessing import Pool

DELIMETER = " "


def _normalize_str(s: str):
    # return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))
    return eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s)))

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
    # print(f"Compare str:\nSTR1: {str1}\nSTR2: {str2}\nR={recall}, P={precision}, F1={f1}")
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

def _clean_node(node_dict):
    def clean_txt(txt):
        return (" ".join([eval_utils.white_space_fix(s).strip() for s in split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], txt) if len(s.strip()) > 0]))
    new_node = {
        'id': node_dict['id'],
        'is_dummy': node_dict['is_dummy'],
        'nondummy_parent': node_dict['nondummy_parent']
    }
    if node_dict['is_dummy'] == False:
        new_node['type'] = node_dict['type']
        if node_dict['type'] == 'text':
            assert node_dict['heading'] == ""
            # new_node['texts'] = node_dict['texts']
            # new_node['texts'] = [eval_utils.white_space_fix(t).strip() for t in node_dict['texts']]
            new_node['texts'] = [clean_txt(t) for t in node_dict['texts']]
        else:
            assert node_dict['type'] in ['head', 'list']
            assert len(node_dict['texts']) == 1
            # new_node['texts'] = [node_dict['heading']]
            # new_node['texts'] = [eval_utils.white_space_fix(node_dict['heading']).strip()]
            new_node['texts'] = [clean_txt(node_dict['heading'])]
    
    return new_node

# true, grobid, shed, llm
def _load_sht(dataset, filename, sht_type):
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

    cleaned_sht_nodes = [_clean_node(n) for n in raw_sht_nodes]
    return cleaned_sht_nodes

def _retrieve_nodes(sht, txt, need_norm):
    retrieved_node_list = []
    for node in sht:
        if node['is_dummy'] == True:
            continue
        norm_txt = txt
        assert [t.strip() == t for t in node['texts']]
        norm_node_text = " ".join(node['texts'])
        if need_norm == True:
            assert False
            norm_txt = _normalize_str(txt)
            norm_node_text = _normalize_str(" ".join(node['texts']))
        if norm_txt.strip() in norm_node_text.strip():
            if node['type'] != 'text':
                retrieved_node_list.append(node['id'])
            else:
                assert node['type'] == 'text'
                parent_id = node['nondummy_parent']
                assert parent_id >= -1
                if parent_id >= 0:
                    parent_node = sht[parent_id]
                    assert parent_node['is_dummy'] == False
                    assert parent_node['type'] != 'text'
                retrieved_node_list.append(parent_id)

    return retrieved_node_list

def _get_node_ctx(sht, node_id):
    if node_id == -1:
        ctx = ""
        last_nid = -1
        for node in sht:
            if node['is_dummy'] == True:
                continue
            assert node['id'] > last_nid
            last_nid = node['id']
            assert isinstance(node['texts'], list)
            for t in node['texts']:
                assert isinstance(t, str)
                assert t.strip() == t
                ctx += t.strip() + " "
        return ctx.strip()
    

    assert sht[node_id]['is_dummy'] == False

    descendant_ids = [node_id]
    flag = False
    last_nid = -1
    for node in sht:
        if node['is_dummy'] == True:
            continue
        assert node['id'] > last_nid
        last_nid = node['id']
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
            assert t.strip() == t
            ctx += t + " "
    return ctx.strip()

def _load_data(true_sht):
    data = []
    for node in true_sht:
        if node['is_dummy'] == True or node['type'] == 'text':
            continue
        assert len(node['texts']) == 1
        tot_text = node['texts'][0]
        sentences = split_text_into_sentences([".", "!", "?", "\n", ",", ";", ":"], tot_text)
        for sid, sentence in enumerate(sentences):
            data.append({
                'id': len(data),
                'txt': sentence,
                'pos': (node['id'], sid)
            })
    assert [d['id'] for d in data] == list(range(len(data)))
    return data

def _comp_score(s1, s2, score_name):
    '''
    Whether s1 is better than s2
    '''
    assert score_name in ['f1', 'recall', 'precision']
    if score_name == "recall":
        if s1['recall'] > s2['recall']:
            return True
        if s1['recall'] == s2['recall'] and s1['f1'] > s2['f1']:
            return True
        return False
    if score_name == "precision":
        if s1['precision'] > s2['precision']:
            return True
        if s1['precision'] == s2['precision'] and s1['f1'] > s2['f1']:
            return True
        return False
    if score_name == "f1":
        if s1['f1'] > s2['f1']:
            return True
        if s1['f1'] == s2['f1'] and s1['recall'] > s2['recall']:
            return True
        return False
    assert False

def _agg_score(data_list):
    # return {
    #     "recall": np.mean([d['score']['recall'] for d in data_list]),
    #     "precision": np.mean([d['score']['precision'] for d in data_list]),
    #     "f1": np.mean([d['score']['f1'] for d in data_list]),
    # }
    m_node_best_score = dict()
    for data_info in data_list:
        node_id = data_info['pos'][0]
        score = data_info['score']
        if node_id not in m_node_best_score:
            m_node_best_score[node_id] = score
        else:
            for score_name in ['f1', 'recall', 'precision']:
                assert score_name in m_node_best_score[node_id]
                if _comp_score(score, m_node_best_score[node_id], score_name):
                    assert score[score_name] >= m_node_best_score[node_id][score_name]
                    m_node_best_score[node_id][score_name] = score[score_name]

    if len(m_node_best_score) == 0:
        assert False

    return {
        "recall": np.mean([m_node_best_score[nid]['recall'] for nid in m_node_best_score]),
        "precision": np.mean([m_node_best_score[nid]['precision'] for nid in m_node_best_score]),
        "f1": np.mean([m_node_best_score[nid]['f1'] for nid in m_node_best_score]),
    }

def _eval_robustness_per_sht(sht, true_sht, need_norm):
    data_list = _load_data(true_sht)
    if len(data_list) == 0:
        return None

    good_node_ids = set()
    for data_info in data_list:
        true_ctx = _get_node_ctx(true_sht, data_info['pos'][0])
        retrieved_node_ids = _retrieve_nodes(sht, data_info['txt'], need_norm=need_norm)
        retrieved_ctxs = [_get_node_ctx(sht, nid) for nid in retrieved_node_ids]
        best_score = {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
        for r_ctx in retrieved_ctxs:
            good_node_ids.add(data_info['pos'][0])
            score = _compare_str(r_ctx, true_ctx)
            for score_name in ['recall', 'precision', 'f1']:
                assert score_name in best_score
                if _comp_score(score, best_score, score_name):
                    assert score[score_name] >= best_score[score_name]
                    best_score[score_name] = score[score_name]
        assert 'score' not in data_info
        data_info['score'] = best_score
    
    tot_node_ids = set([d['pos'][0] for d in data_list])
    assert good_node_ids.issubset(tot_node_ids)
    print(f"Good nodes: {len(good_node_ids)}/{len(tot_node_ids)}")
    print(f"Bad nodes: {sorted(list(tot_node_ids - good_node_ids))}")
    return _agg_score(data_list)

def eval_top_down_robustness(dataset, sht_type):
    m_filename_score = dict()
    for pdf_name in sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf'))):
        filename = pdf_name.replace(".pdf", "")
        true_sht = _load_sht(dataset, filename, 'intrinsic')
        sht = _load_sht(dataset, filename, sht_type)
        score = _eval_robustness_per_sht(sht, true_sht, need_norm=False)
        if score == None:
            print(f"[{dataset}][{sht_type}] {filename}: No non-dummy non-text nodes in true SHT, skip.")
            continue
        m_filename_score[filename] = score
        print(f"[{dataset}][{sht_type}] {filename}: R={score['recall']:.4f}, P={score['precision']:.4f}, F1={score['f1']:.4f}")
    
    overall_score = {
        "recall": np.mean([m_filename_score[f]['recall'] for f in m_filename_score]),
        "precision": np.mean([m_filename_score[f]['precision'] for f in m_filename_score]),
        "f1": np.mean([m_filename_score[f]['f1'] for f in m_filename_score]),
    }

    return overall_score

def _eval_robustness_per_sht_parallel(tuple_args):
    sht, true_sht, need_norm = tuple_args
    return _eval_robustness_per_sht(sht, true_sht, need_norm)

def eval_top_down_robustness_parallel(dataset, sht_type):
    m_filename_score = dict()
    tuple_args_list = []
    filename_list = []
    for pdf_name in sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf'))):
        filename = pdf_name.replace(".pdf", "")
        true_sht = _load_sht(dataset, filename, 'intrinsic')
        sht = _load_sht(dataset, filename, sht_type)
        tuple_args_list.append((sht, true_sht, False))
        filename_list.append(filename)

    with Pool(processes=(len(tuple_args_list) + 2)) as pool:
        results = pool.map(_eval_robustness_per_sht_parallel, tuple_args_list)

    for filename, score in zip(filename_list, results):
        if score == None:
            print(f"[{dataset}][{sht_type}] {filename}: No text nodes in true SHT, skipping...")
            continue
        m_filename_score[filename] = score
        print(f"[{dataset}][{sht_type}] {filename}: R={score['recall']:.4f}, P={score['precision']:.4f}, F1={score['f1']:.4f}")
    
    overall_score = {
        "recall": np.mean([m_filename_score[f]['recall'] for f in m_filename_score]),
        "precision": np.mean([m_filename_score[f]['precision'] for f in m_filename_score]),
        "f1": np.mean([m_filename_score[f]['f1'] for f in m_filename_score]),
    }

    return overall_score

if __name__ == "__main__":
    import time
    logging.disable(logging.INFO)
    # method_list = ['llm_txt', 'grobid', 'wide', 'deep', 'shed']
    # method_list = ['llm_txt', 'grobid', 'wide', 'shed']
    method_list = ['llm_vision']
    dataset_list = ['civic', 'contract', 'qasper', 'finance']

    tot_tab_str = ""
    for sht_type in method_list:
        tab_str = f"{sht_type}"
        m_dataset_score = dict()
        for dataset in dataset_list:
            start = time.time()
            overall_score = eval_top_down_robustness_parallel(dataset, sht_type)
            end = time.time()
            print(end-start)
            m_dataset_score[dataset] = overall_score
            print(f"[{sht_type}][{dataset}] Overall: R={overall_score['recall']:.4f}, P={overall_score['precision']:.4f}, F1={overall_score['f1']:.4f}")
        for score_name in ['recall', 'precision', 'f1']:
            for dataset in dataset_list:
                tab_str += f" & {m_dataset_score[dataset][score_name]:.4f}"
        print(tab_str)
        tot_tab_str += tab_str + " \\\\\n"

    print(tot_tab_str)
        