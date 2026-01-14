from structured_rag import StructuredRAG, get_context_len
import argparse
import os
import json
import logging
import logging_config
import traceback


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no).")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root-dir", type=str, required=True, help="root dir (pdf file is under root_dir/pdf/)")
    # parser.add_argument("--chunk-size", type=int, required=False, default=100, help="size of a chunk (i.e., the length of the context of a newly added leaf)")
    # parser.add_argument("--summary-len", type=int, required=False, default=100, help="length of a recursively generated summary (i.e., the length of the context of an original SHT node)")
    # parser.add_argument("--node-embedding-model", type=str, required=False, choices=["sbert", "dpr", "te3small"], default="sbert", help="the embedding model for SHT nodes")
    # parser.add_argument("--query-embedding-model", type=str, required=False, choices=["sbert", "dpr", "te3small"], default="sbert", help="the embedding model for the query")
    # parser.add_argument("--summarization-model", type=str, required=False, choices=["gpt-4o-mini", "empty"], default="gpt-4o-mini", help="the summarization model")
    # parser.add_argument("--embed-hierarchy", type=str_to_bool, required=False, default=True, help="whether to embed the hierarchical information")
    # parser.add_argument("--distance-metric", type=str, choices=["cosine", "L1", "L2", "Linf"], required=False, default="cosine", help="the distance metric in the embedding space")
    # parser.add_argument("--context-hierarchy", type=str_to_bool, required=False, default=True, help="whether to recover hierarchical information in the final context")
    # parser.add_argument("--context-raw", type=str_to_bool, required=False, default=True, help="whether to retrieve the newly added leaves (i.e., the chunks of the document) for the final context")
    # parser.add_argument("--context-len", type=int, required=False, default=1000, help='the length of the final context')
    
    # args = parser.parse_args()

    # print(
    #     args.root_dir,
    #     args.chunk_size, 
    #     args.summary_len, 
    #     args.node_embedding_model, 
    #     args.query_embedding_model, 
    #     args.summarization_model,
    #     args.embed_hierarchy,
    #     args.distance_metric,
    #     args.context_hierarchy,
    #     args.context_raw,
    #     args.context_len
    # )

    # queries_path = os.path.join(args.root_dir, "queries.json")
    # with open(queries_path, 'r') as file:
    #     queries_info = json.load(file)
    
    # for qid, query_info in enumerate(queries_info):
    #     assert qid == query_info["id"]
    #     rag = StructuredRAG(
    #         root_dir=args.root_dir,
    #         chunk_size=args.chunk_size,
    #         summary_len=args.summary_len,
    #         node_embedding_model=args.node_embedding_model,
    #         query_embedding_model=args.query_embedding_model,
    #         summarization_model=args.summarization_model,
    #         embed_hierarchy=args.embed_hierarchy,
    #         distance_metric=args.distance_metric,
    #         context_hierarchy=args.context_hierarchy,
    #         context_raw=args.context_raw,
    #         context_len=args.context_len
    #     )
    #     rag.generate_context(
    #         name=query_info["file_name"],
    #         query=query_info["query"],
    #         query_id=qid,
    #     )



    ############################### build sht
    # import config
    # from datetime import datetime
    # dataset = "finance"
    # root_dir = os.path.join(config.DATA_ROOT_FOLDER, dataset)
    # assert os.path.exists(root_dir)

    # for pdf_filename in sorted(os.listdir(os.path.join(root_dir, "pdf"))):
    #     pdf_path = os.path.join(root_dir, "pdf", pdf_filename)
    #     assert os.path.exists(pdf_path)
    #     cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     print(f"[{cur_time}] Processing {pdf_filename}...")
    #     try:
    #         # for node_embedding_model in ["sbert", "dpr", "te3small"]:
    #         for node_embedding_model in ["dpr", "te3small"]:
    #             rag = StructuredRAG(
    #                 root_dir=root_dir,
    #                 chunk_size=100,
    #                 summary_len=100,
    #                 node_embedding_model=node_embedding_model,
    #                 query_embedding_model=node_embedding_model,
    #                 summarization_model="gpt-4o-mini",
    #                 embed_hierarchy=True,
    #                 distance_metric="cosine",
    #                 context_hierarchy=True,
    #                 context_raw=True,
    #                 context_len=0.05
    #             )
    #             rag.build_sht(
    #                 name=pdf_filename.replace(".pdf", "")
    #             )
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print(f"\t[{cur_time}]✅ Finished processing {pdf_filename} with {node_embedding_model}")
    #     except Exception as e:
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]❌ Error processing {pdf_filename}: {e}")
    #         continue

    ################################################ gen index
    import config
    from datetime import datetime
    is_true_sht = True
    dataset = "finance"
    root_dir = os.path.join(config.DATA_ROOT_FOLDER, dataset)
    assert os.path.exists(root_dir)
    if is_true_sht == True:
        root_dir = os.path.join(root_dir, "intrinsic")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json"), 'r') as file:
        qinfo_list = json.load(file)
    
    for embed_hierarchy in [
        True, 
        # False
    ]:
        print(f"embed_hierarchy={embed_hierarchy}")
        for qinfo in qinfo_list:
            file_name = qinfo["file_name"]
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{cur_time}] Processing {qinfo['id']}...")
            try:
                # node_embedding_model_list = ["sbert", "dpr", "te3small"]
                node_embedding_model_list = ["sbert"]
                if embed_hierarchy == False:
                    node_embedding_model_list = ["sbert"]
                for node_embedding_model in node_embedding_model_list:
                    if is_true_sht == False:
                        sht_path = os.path.join(root_dir, f"{node_embedding_model}.gpt-4o-mini.c100.s100", "sht", f"{file_name}.json")
                    else:
                        sht_path = os.path.join(root_dir, "intrinsic", f"{node_embedding_model}.gpt-4o-mini.c100.s100", "sht", f"{file_name}.json")
                    assert os.path.exists(sht_path), f"No SHT: {sht_path} doesn't exist"
                    rag = StructuredRAG(
                        root_dir=root_dir,
                        chunk_size=100,
                        summary_len=100,
                        node_embedding_model=node_embedding_model,
                        query_embedding_model=node_embedding_model,
                        summarization_model="gpt-4o-mini",
                        embed_hierarchy=embed_hierarchy,
                        distance_metric="cosine",
                        context_hierarchy=True,
                        context_raw=True,
                        context_len=0.05
                    )
                    rag.index(
                        name=file_name,
                        query=qinfo["query"],
                        query_id=qinfo["id"]
                    )
                    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\t[{cur_time}]✅ Finished processing {qinfo['id']} with {node_embedding_model}")
            except Exception as e:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]❌ Error processing {qinfo['id']}: {e}\n{traceback.print_exc()}")
                continue