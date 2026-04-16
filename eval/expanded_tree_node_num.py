import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raptor.tree_structures import Tree, Node
import pickle


def _estimate_raptor_node(raptor_tree: Tree):
    return len(raptor_tree.all_nodes)

def _estimate_shed_node(sht):
    all_nodes = sht['nodes']
    expanded_node_cnt = 0
    for node in all_nodes:
        if node['is_dummy'] == True:
            continue
        if node['type'] == 'text':
            expanded_node_cnt += len(node['texts'])
        else:
            assert node['type'] in ['head', 'list']
            expanded_node_cnt += 1
    return expanded_node_cnt


def per_dataset_avg_node(dataset, sht_type):
    if sht_type == "shed":
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht")
        load_tree = lambda sht_path: json.load(open(sht_path, 'r'))
        get_node_cnt = lambda sht: _estimate_shed_node(sht)
    elif sht_type == "raptor":
        folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "baselines", "raptor_tree")
        load_tree = lambda sht_path: pickle.load(open(sht_path, 'rb'))
        get_node_cnt = lambda sht: _estimate_raptor_node(sht)
    else:
        raise ValueError(f"Unknown sht_type: {sht_type}")

    total_node = 0.0
    for sht_name in os.listdir(folder):
        sht_path = os.path.join(folder, sht_name)
        sht = load_tree(sht_path)
        node_cnt = get_node_cnt(sht)
        total_node += node_cnt

    avg_node = total_node / len(os.listdir(folder))
    return avg_node

        

if __name__ == "__main__":
    # sht_type_list = ['shed', 'raptor']
    # dataset_list = [
    #     'civic', 
    #     'contract', 
    #     'qasper', 
    #     'finance'
    #     ]

    # for dataset in dataset_list:
    #     for sht_type in sht_type_list:
    #         avg_node = per_dataset_avg_node(dataset, sht_type)
    #         print(f"dataset={dataset}, sht_type={sht_type}, avg_node={avg_node}")


    # dataset=civic, sht_type=shed, avg_node=172.3684210526316
    # dataset=civic, sht_type=raptor, avg_node=37.63157894736842
    # dataset=contract, sht_type=shed, avg_node=53.10958904109589
    # dataset=contract, sht_type=raptor, avg_node=30.17808219178082
    # dataset=qasper, sht_type=shed, avg_node=170.06730769230768
    # dataset=qasper, sht_type=raptor, avg_node=124.42548076923077
    # dataset=finance, sht_type=shed, avg_node=1869.857142857143
    # dataset=finance, sht_type=raptor, avg_node=1546.797619047619


    shed_tot_nodes = round(172.37 * 19) + round(53.11 * 73) + round(170.07 * 416) + round(1869.86 * 84)
    raptor_tot_nodes = round(37.63 * 19) + round(30.18 * 73) + round(124.43 * 416) + round(1546.80 * 84)

    print(shed_tot_nodes/raptor_tot_nodes)