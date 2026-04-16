import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
from structured_rag import utils, split_text_into_sentences
import numpy as np

def _clean_node(node_dict):
    new_node = {
        'id': node_dict['id'],
        'is_dummy': node_dict['is_dummy'],
        'nondummy_parent': node_dict['nondummy_parent']
    }
    if node_dict['is_dummy'] == False:
        new_node['type'] = node_dict['type']
        if node_dict['type'] == 'text':
            assert node_dict['heading'] == ""
            new_node['texts'] = [t for t in node_dict['texts']]
        else:
            assert node_dict['type'] in ['head', 'list']
            assert len(node_dict['texts']) == 1
            new_node['texts'] = [node_dict['heading']]
    
    return new_node

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
    
    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    cleaned_sht_nodes = [_clean_node(n) for n in raw_sht_nodes]
    return cleaned_sht_nodes

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
            ctx += t + " "
    return ctx.strip()


def comp_compactness(dataset):
    sht_dir = f"/home/ruiying/SHTRAG/data/{dataset}/sbert.gpt-4o-mini.c100.s100/sht"

    file_name_list = sorted(os.listdir(sht_dir))

    ratio_list = []

    for file_name in file_name_list:
        true_sht = _load_sht(dataset, file_name.replace(".json", ""), sht_type='true')
        sht = _load_sht(dataset, file_name.replace(".json", ""), sht_type='shed')
        deep_sht = _load_sht(dataset, file_name.replace(".json", ""), sht_type='deep')

        sht_path = os.path.join(sht_dir, file_name)
        # with open(sht_path, 'r') as f:
        #     doc = json.load(f)['full_text']

        # doc_len = len(doc.split())
        

        diff_compactness = 0
        sht_nodes = [n for n in sht if n['is_dummy'] == False and n['type'] in ['head', 'list']]
        deep_sht_nodes = [n for n in deep_sht if n['is_dummy'] == False and n['type'] in ['head', 'list']]
        true_sht_nodes = [n for n in true_sht if n['is_dummy'] == False and n['type'] in ['head', 'list']]
        assert sorted(n['id'] for n in sht_nodes) == [n['id'] for n in sht_nodes]
        assert sorted(n['id'] for n in deep_sht_nodes) == [n['id'] for n in deep_sht_nodes]
        assert sorted(n['id'] for n in true_sht_nodes) == [n['id'] for n in true_sht_nodes]
        assert len(sht_nodes) == len(deep_sht_nodes)
        for sht_node, deep_sht_node in zip(sht_nodes, deep_sht_nodes):
            assert sht_node['type'] == deep_sht_node['type']
            sht_ctx = _get_node_ctx(sht, sht_node['id'])
            deep_sht_ctx = _get_node_ctx(deep_sht, deep_sht_node['id'])
            sht_ctx_len = len(sht_ctx.split())
            deep_sht_ctx_len = len(deep_sht_ctx.split())
            diff_compactness += (sht_ctx_len - deep_sht_ctx_len) / doc_len
        
        ratio_list.append(diff_compactness)
    

    avg_ratio = sum(ratio_list) / len(ratio_list)
    print(f"Average compactness improvement from SHT to deep SHT for dataset {dataset}: {avg_ratio:.4f}")
    return avg_ratio

if __name__ == "__main__":
    comp_list = []
    for dataset in ['civic', 'contract', 'qasper', 'finance']:
        comp_list.append(comp_compactness(dataset))

    # overall_comp = np.mean(comp_list)
    overall_comp = (comp_list[0] * 19 + comp_list[1] * 73 + comp_list[2] * 416 + comp_list[3] * 84) / (19 + 73 + 416 + 84)
    print(f"Overall average compactness improvement from SHT to deep SHT: {overall_comp:.4f}")