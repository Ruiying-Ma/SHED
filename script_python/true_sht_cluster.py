import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import traceback
import logging
import logging_config

def gen_true_clustering(csv_path, orig_cluster_json_path, dst_cluster_json_path):
    print(csv_path)
    with open(csv_path, 'r') as file:
        header_lines = [l.strip() for l in file.readlines() if len(l.strip()) > 0]
    
    with open(orig_cluster_json_path, 'r') as file:
        orig_items = json.load(file)

    true_header_dict = dict()
    for hl in header_lines:
        id = None
        try:
            id = int(hl.strip().split(",")[-1])
        except:
            traceback.print_exc()
        assert id != None, hl
        assert id not in true_header_dict, f"{id}, {true_header_dict}"
        true_header_dict[id] = ",".join(hl.strip().split(",")[1:-1])
    

    orig_header_dict = dict()
    for oi in orig_items:
        id = oi['id']
        assert id not in orig_header_dict
        orig_header_dict[id] = oi

    def load_true_header(h):
        json_str = "{" + f'''"h": "{h}"''' + "}"
        try:
            return json.loads(json_str)["h"].strip()
        except:
            print(h)
            traceback.print_exc()
        assert False

    # check
    for id in true_header_dict:
        true_h = load_true_header(true_header_dict[id])
        assert orig_header_dict[id]['text'].count(true_h) == 1, f"{csv_path}\n\tid: {id}\n\ttrue: {true_h} ({true_header_dict[id]})\n\torig: {orig_header_dict[id]['text']}"
    
    # get cluster id
    def get_level(h: str):
        expanded = h.expandtabs(4)
        return len(expanded) - len(expanded.lstrip(' '))
    cluster_dict = dict()
    for id in sorted(list(true_header_dict.keys())):
        assert id not in cluster_dict
        true_header = true_header_dict[id]
        if len(cluster_dict) == 0:
            assert get_level(true_header) == 0
            cluster_dict[id] = -1
            continue
        assert all([i < id for i in cluster_dict])
        cur_level = get_level(true_header)
        parent_id = None
        for existed_id in sorted(list(cluster_dict.keys()), reverse=True):
            if cur_level > get_level(true_header_dict[existed_id]):
                parent_id = existed_id
                break
        assert parent_id != None, true_header
        cluster_dict[id] = cluster_dict[parent_id] + 1

    assert set(cluster_dict.keys()) == set(true_header_dict.keys())
    
    # reconstruct items
    new_item_list = []
    for id in sorted(list(orig_header_dict.keys())):
        item = orig_header_dict[id]
        if id not in cluster_dict:
            item['type'] = 'Text'
            item['id'] = len(new_item_list)
            item['features'] = dict()
            item.pop('cluster_id', None)
            new_item_list.append(item)
        else:
            # true_header = true_header_dict[id].strip()
            true_header = load_true_header(true_header_dict[id])
            cluster_id = cluster_dict[id]
            if true_header == item['text']:
                item['features'] = dict()
                item['cluster_id'] = cluster_id
                item['type'] = 'Section header'
                item['id'] = len(new_item_list)
                new_item_list.append(item)
            else:
                assert item['text'].count(true_header) == 1
                start_id = item['text'].find(true_header)
                before_text = item['text'][:start_id]
                after_text = item['text'][start_id + len(true_header):]
                assert before_text + true_header + after_text == item['text']
                before_item = {k: v for k, v in item.items() if k not in ['cluster_id']}
                cur_item = {k: v for k, v in item.items() if k != "features"}
                after_item = {k: v for k, v in item.items() if k not in ['cluster_id']}
                before_item['text'] = before_text
                cur_item['text'] = true_header
                after_item['text'] = after_text
                before_item['type'] = 'Text'
                cur_item['type'] = 'Section header'
                after_item['type'] = 'Text'
                before_item['features'] = dict()
                cur_item['features'] = dict()
                after_item['features'] = dict()
                cur_item['cluster_id'] = cluster_id
                
                if len(before_item['text']) != 0:
                    before_item['id'] = len(new_item_list)
                    new_item_list.append(before_item)
                cur_item['id'] = len(new_item_list)
                new_item_list.append(cur_item)
                if len(after_item['text']) != 0:
                    after_item['id'] = len(new_item_list)
                    new_item_list.append(after_item)

    os.makedirs(os.path.dirname(dst_cluster_json_path), exist_ok=True)
    with open(dst_cluster_json_path, 'w') as file:
        json.dump(new_item_list, file, indent=4)


def grobid2sht(dataset):
    grobid_dir = os.path.join("/home/ruiying/SHTRAG/data", dataset, "intrinsic")
    grobid_clustering_dir = os.path.join(grobid_dir, "node_clustering")
    os.makedirs(grobid_clustering_dir, exist_ok=True)
    sht_skeleton_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht_skeleton")
    sht_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht")
    sht_vis_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht_vis")
    os.makedirs(sht_skeleton_dir, exist_ok=True)
    os.makedirs(sht_dir, exist_ok=True)
    os.makedirs(sht_vis_dir, exist_ok=True)
    print(f"{len(os.listdir(grobid_clustering_dir))} true SHTs...")

    # build_tree_skeleton
    for json_name in sorted(os.listdir(grobid_clustering_dir)):
        if os.path.exists(os.path.join(sht_skeleton_dir, json_name)):
            print(f"SHT skeleton for {json_name} is already existed.")
            continue
        print("Building SHT skeleton for", json_name)
        assert json_name.endswith(".json")
        cluster_jsonl_path = os.path.join(grobid_clustering_dir, json_name)
        assert os.path.exists(cluster_jsonl_path), cluster_jsonl_path
        with open(cluster_jsonl_path, 'r') as file:
            objects = json.load(file)
        sht_builder = SHTBuilder(
            config=SHTBuilderConfig(
                store_json=os.path.join(sht_skeleton_dir, json_name),
                load_json=None,
                chunk_size=100,
                summary_len=100,
                embedding_model_name="sbert",
                summarization_model_name="gpt-4o-mini",
            )
        )
        logging.info(f"Built SHT skeleton for {json_name}")
        sht_builder.build(objects)
        sht_builder.check()
        sht_builder.store2json()
        sht_builder.visualize(vis_path=os.path.join(sht_vis_dir, json_name.replace(".json", ".vis")))

    # # add summaries and embeddings
    # for json_name in sorted(os.listdir(grobid_clustering_dir)):
    #     try:
    #         assert json_name.endswith(".json")
    #         sht_skeleton_path = os.path.join(sht_skeleton_dir, json_name)
    #         sht_path = os.path.join(sht_dir, json_name)
    #         if os.path.exists(sht_path):
    #             print(f"SHT (complete) for {json_name} is already existed")
    #             continue
    #         sht_builder = SHTBuilder(
    #             config=SHTBuilderConfig(
    #                 store_json=sht_path,
    #                 load_json=sht_skeleton_path,
    #                 chunk_size=100,
    #                 summary_len=100,
    #                 embedding_model_name="sbert",
    #                 summarization_model_name="gpt-4o-mini",
    #             )
    #         )
    #         logging.info(f"Processing {json_name} with sbert")
    #         sht_builder.build(None)
    #         sht_builder.check()
    #         logging.info(f"\tAdding summaries for {json_name}")
    #         sht_builder.add_summaries()
    #         sht_builder.check()
    #         node_ids = list(range(len(sht_builder.tree["nodes"])))
    #         logging.info(f"\tAdding embeddings for {json_name}")
    #         sht_builder.add_embeddings(node_ids)
    #         sht_builder.store2json()
    #         with open(sht_builder.store_json, 'w') as file:
    #             json.dump(sht_builder.tree, file, indent=4)

    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]✅ Finished processing {json_name} with sbert")
    #     except Exception as e:
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]❌ Error processing {json_name} with sbert: {str(e)}\n{traceback.print_exc()}")
    #         continue


    # # index / generate context for queries
    # root_dir = os.path.dirname(grobid_dir)
    # queries_path = os.path.join(root_dir, "queries.json")
    # with open(queries_path, 'r') as file:
    #     queries_info = json.load(file)

    # for qid, query_info in enumerate(queries_info):
    #     file_name = query_info["file_name"]
    #     cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     print(f"[{cur_time}] Processing {qid}...")
    #     try:
    #         sht_path = f"/home/ruiying/SHTRAG/data/{dataset}/grobid/sbert.gpt-4o-mini.c100.s100/sht/{file_name}.json"
    #         assert os.path.exists(sht_path), f"No SHT: {sht_path} doesn't exist"
    #         rag = StructuredRAG(
    #             root_dir=grobid_dir,
    #             chunk_size=100,
    #             summary_len=100,
    #             node_embedding_model="sbert",
    #             query_embedding_model="sbert",
    #             summarization_model="gpt-4o-mini",
    #             embed_hierarchy=True,
    #             distance_metric="cosine",
    #             context_hierarchy=True,
    #             context_raw=True,
    #             context_len=1000
    #         )
    #         ### index
    #         rag.index(
    #             name=query_info["file_name"],
    #             query=query_info["query"],
    #             query_id=qid,
    #         )
    #         ### generate context
    #         # rag.generate_context(
    #         #     name=query_info["file_name"],
    #         #     query=query_info["query"],
    #         #     query_id=qid,
    #         # )
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]✅ Finished processing {qid} with sbert")
    #     except Exception as e:
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]❌ Error processing {qid}: {e}\n{traceback.print_exc()}")
    #         continue

if __name__ == "__main__":
    # logging.disable(level=logging.DEBUG)
    # #############################################gen clustering
    # csv_folder = "/home/ruiying/SHTRAG/data/finance/intrinsic/human_label"
    # orig_cluster_folder = "/home/ruiying/SHTRAG/data/finance/node_clustering"
    # true_cluster_folder = "/home/ruiying/SHTRAG/data/finance/intrinsic/node_clustering"
    # for csvname in sorted(os.listdir(csv_folder)):
    #     if csvname == "VERIZON_2022_10K.csv":

    #         filename = csvname.replace(".csv", "")
    #         gen_true_clustering(
    #             csv_path=os.path.join(csv_folder, csvname),
    #             orig_cluster_json_path=os.path.join(orig_cluster_folder, filename+".json"),
    #             dst_cluster_json_path=os.path.join(true_cluster_folder, filename+".json")
    #         )
    
    # #########################################build SHT
    grobid2sht('civic')

    

