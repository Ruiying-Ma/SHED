import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import traceback
import logging
import logging_config

def gen_wide_clustering(orig_cluster_json_path, dst_cluster_json_path):
    with open(orig_cluster_json_path, 'r') as file:
        orig_items = json.load(file)
    # build clustering
    is_first = True
    for oi in orig_items:
        if 'cluster_id' in oi and is_first == True:
            oi['cluster_id'] = 0
            is_first = False
            continue
        if 'cluster_id' in oi and is_first == False:
            oi['cluster_id'] = 1
    os.makedirs(os.path.dirname(dst_cluster_json_path), exist_ok=True)
    with open(dst_cluster_json_path, 'w') as file:
        json.dump(orig_items, file, indent=4)


def grobid2sht(dataset):
    grobid_dir = os.path.join("/home/ruiying/SHTRAG/data", dataset, "wide")
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
        sht_builder.build(objects, 'wide')
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
    #         sht_builder.build(None, 'wide')
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
    #         sht_path = f"/home/ruiying/SHTRAG/data/{dataset}/wide/sbert.gpt-4o-mini.c100.s100/sht/{file_name}.json"
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
    logging.disable(level=logging.DEBUG)
    #############################################gen clustering
    # for dataset in ['civic', 'contract', 'finance', 'qasper']:
    for dataset in ['office']:
        # orig_cluster_folder = f"/home/ruiying/SHTRAG/data/{dataset}/node_clustering"
        # wide_cluster_folder = f"/home/ruiying/SHTRAG/data/{dataset}/wide/node_clustering"
        # os.makedirs(wide_cluster_folder, exist_ok=True)
        # for jsonname in sorted(os.listdir(orig_cluster_folder)):
        #     filename = jsonname.replace(".json", "")
        #     gen_wide_clustering(
        #         orig_cluster_json_path=os.path.join(orig_cluster_folder, filename+".json"),
        #         dst_cluster_json_path=os.path.join(wide_cluster_folder, filename+".json")
        #     )
        
        # #########################################build SHT
        grobid2sht(dataset)

    

