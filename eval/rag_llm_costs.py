import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raptor.tree_structures import Tree, Node
import tiktoken
import pickle

tokenizer = tiktoken.get_encoding('cl100k_base')

def _estimate_raptor_cost(raptor_tree: Tree, model):
    total_input_tokens = 0
    total_output_tokens = 0

    for node_idx, node in raptor_tree.all_nodes.items():
        assert isinstance(node, Node)
        assert isinstance(node_idx, int)
        assert node.index == node_idx
        has_children = len(node.children) > 0
        txt_token_num = len(tokenizer.encode(node.text + "\n\n"))
        if has_children == True:
            total_output_tokens += txt_token_num
            for child_idx in node.children:
                child_node = raptor_tree.all_nodes[child_idx]
                assert isinstance(child_node, Node)
                assert child_node.index == child_idx
                child_txt_token_num = len(tokenizer.encode(child_node.text + "\n\n"))
                total_input_tokens += child_txt_token_num

    return _cost(total_input_tokens, total_output_tokens, model)

def _estimated_shed_cost(sht, model):
    tot_input_tokens = sht['estimated_cost']['input_tokens']
    tot_output_tokens = sht['estimated_cost']['output_tokens']
    return _cost(tot_input_tokens, tot_output_tokens, model)

def _cost(input_tokens, output_tokens, model):
    if model == "gpt-4o-mini":
        return (input_tokens * 0.15 + output_tokens * 0.6) / 1000000
    else:
        raise ValueError(f"Unknown model: {model}")
    

def per_dataset_cost(dataset, sht_type, model="gpt-4o-mini"):
    if sht_type == "shed":
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht")
        load_tree = lambda sht_path: json.load(open(sht_path, 'r'))
        get_cost = lambda sht: _estimated_shed_cost(sht, model)
    elif sht_type == "raptor":
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "baselines", "raptor_tree")
        load_tree = lambda sht_path: pickle.load(open(sht_path, 'rb'))
        get_cost = lambda sht: _estimate_raptor_cost(sht, model)
    else:
        raise ValueError(f"Unknown sht_type: {sht_type}")

    total_cost = 0.0
    for sht_name in os.listdir(folder):
        sht_path = os.path.join(folder, sht_name)
        sht = load_tree(sht_path)
        cost = get_cost(sht)
        total_cost += cost

    # print(f"Total estimated cost for dataset '{dataset}': ${total_cost:.6f}")
    return total_cost
        

if __name__ == "__main__":
    sht_type_list = ['shed', 'raptor']
    dataset_list = [
        'civic', 
        'contract', 
        'qasper', 
        'finance'
        ]

    for dataset in dataset_list:
        for sht_type in sht_type_list:
            tot_cost = per_dataset_cost(dataset, sht_type, model="gpt-4o-mini")
            print(f"Dataset: {dataset}, SHT Type: {sht_type}, Total Cost: ${tot_cost:.6f}")