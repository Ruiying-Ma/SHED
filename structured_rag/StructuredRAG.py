import subprocess
import json
from .ClusteringOracle import ClusteringOracle, ClusteringOracleConfig
from .SHTBuilder import SHTBuilder, SHTBuilderConfig
from .SHTIndexer import SHTIndexerConfig, SHTIndexer
from .SHTGenerator import SHTGeneratorConfig, SHTGenerator
import os
import logging
from .utils import get_context_len

def _write_to_file(dest_path, contents, is_json=False, is_append=False):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if is_json:
        # print(f"write_to_file: {dest_path}")
        assert is_append == False
        assert dest_path.endswith(".json")
        with open(dest_path, 'w') as file:
            json.dump(contents, file, indent=4)
        return
    if is_append:
        # print(f"append_to_file: {dest_path}")
        assert is_json == False
        with open(dest_path, 'a') as file:
            file.write(contents)

class StructuredRAG:
    def __init__(
        self, 
        root_dir,
        chunk_size, 
        summary_len, 
        node_embedding_model, 
        query_embedding_model, 
        summarization_model,
        embed_hierarchy,
        distance_metric,
        context_hierarchy,
        context_raw,
        context_len
    ):
        self.heading_identification_dir = os.path.join(root_dir, "heading_identification")
        self.node_clustering_dir = os.path.join(root_dir, "node_clustering")
        self.pdf_dir = os.path.join(root_dir, "pdf")

        sub_root_dir = os.path.join(root_dir, f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}") # determined by SHTBuilderConfig
        self.sht_dir =  os.path.join(sub_root_dir, "sht")
        self.sht_vis_dir = os.path.join(sub_root_dir, "sht_vis")
        
        sub_sub_root_dir = os.path.join(sub_root_dir, f"{query_embedding_model}.{distance_metric}.h{int(embed_hierarchy)}")
        self.index_path = os.path.join(sub_sub_root_dir, "index.jsonl")

        sub_sub_sub_root_dir = os.path.join(sub_sub_root_dir, f"l{int(context_raw)}.h{int(context_hierarchy)}", f"context{context_len}")
        self.context_path = os.path.join(sub_sub_sub_root_dir, "context.jsonl")

        os.makedirs(self.node_clustering_dir, exist_ok=True)
        os.makedirs(self.sht_dir, exist_ok=True)
        os.makedirs(self.sht_vis_dir, exist_ok=True)


        # stats
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_time = 0.0
        self.embedding_time = {
            "hybrid": 0.0,
            "texts": 0.0,
            "heading": 0.0
        }

        # SHTBuilder
        self.chunk_size = chunk_size
        self.summary_len = summary_len
        self.node_embedding_model = node_embedding_model
        self.summarization_model = summarization_model

        # SHTIndexer
        self.query_embedding_model = query_embedding_model
        self.embed_hierarchy = embed_hierarchy
        self.distance_metric = distance_metric

        # SHTGenerator
        self.context_hierarchy = context_hierarchy
        self.context_raw = context_raw
        self.context_len = context_len


    def heading_indentification(self, name):
        pdf_path = os.path.join(self.pdf_dir, name+".pdf")
        assert os.path.exists(pdf_path)
        heading_identification_path = os.path.join(self.heading_identification_dir, name+".json")
        if os.path.exists(heading_identification_path):
            logging.debug(f"Heading identification already existed: {heading_identification_path}!")
            with open(heading_identification_path, 'r') as file:
                result = json.load(file)
            return result
        logging.debug(f"Doing heading identification: {heading_identification_path}...")
        curl_command = f'''curl -X POST -F 'file=@{pdf_path}' localhost:5060'''
        result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
        _write_to_file(
            dest_path=heading_identification_path,
            contents=result.stdout,
            is_json=True,
        )
        return result.stdout
    
    def node_clustering(self, name):
        node_clustering_path = os.path.join(self.node_clustering_dir, name+".json")
        if os.path.exists(node_clustering_path):
            logging.debug(f"Node clustering already existed: {node_clustering_path}!")
            with open(node_clustering_path, 'r') as file:
                new_object_dicts_list = json.load(file)
                return new_object_dicts_list
        
        logging.debug(f"Doing node clustering: {node_clustering_path}...")
        object_dicts_list = self.heading_indentification(name)

        clustering_oracle = ClusteringOracle(
            config=ClusteringOracleConfig(store_json=node_clustering_path)
        )
        new_object_dicts_list = clustering_oracle.cluster(
            pdf_path=os.path.join(self.pdf_dir, name+".pdf"),
            object_dicts_list=object_dicts_list
        )
        return new_object_dicts_list

    def build_sht(self, name):
        sht_path = os.path.join(self.sht_dir, name+".json")
        if os.path.exists(sht_path):
            logging.debug(f"SHT already existed: {sht_path}!")
            with open(sht_path, 'r') as file:
                sht = json.load(file)
                return sht
        else:
            logging.debug(f"Building SHT: {sht_path}...")
            sht_load_path = None
            assert sht_path.count(self.node_embedding_model) == 1
            candid_embedding_models = ["sbert", "dpr", "te3small"]
            for candid_embedding_model in candid_embedding_models:
                candid_sht_load_path = sht_path.replace(self.node_embedding_model, candid_embedding_model)
                if os.path.exists(candid_sht_load_path):
                    sht_load_path = candid_sht_load_path
                    break
            if sht_load_path != None:
                new_object_dicts_list = None
            else:
                new_object_dicts_list = self.node_clustering(name)


        sht_builder = SHTBuilder(
            config=SHTBuilderConfig(
                store_json=sht_path,
                load_json=sht_load_path,
                chunk_size=self.chunk_size,
                summary_len=self.summary_len,
                embedding_model_name=self.node_embedding_model,
                summarization_model_name=self.summarization_model,
            )
        )

        sht_builder.build(new_object_dicts_list)
        sht_builder.check()
        sum_stats = None
        if sht_load_path == None:
            sum_stats = sht_builder.add_summaries()
            sht_builder.check()
        node_ids = list(range(len(sht_builder.tree["nodes"])))
        embed_stats = sht_builder.add_embeddings(node_ids)
        sht_builder.check()

        sht_builder.store2json()
        if sht_load_path == None:
            sht_builder.visualize(vis_path=os.path.join(self.sht_vis_dir, name+".vis"))

        if sum_stats != None:
            self.input_tokens += sum_stats["input_tokens"]
            self.output_tokens += sum_stats["output_tokens"]
            self.llm_time += sum_stats["time"]
        self.embedding_time["hybrid"] += embed_stats["hybrid"]
        self.embedding_time["texts"] += embed_stats["texts"]
        self.embedding_time["heading"] += embed_stats["heading"]
        
        return sht_builder.tree

    def index(self, name, query, query_id):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as file:
                for l in file:
                    index_info = json.loads(l)
                    if int(index_info["id"]) == int(query_id):
                        logging.debug(f"Index already existed: {self.index_path}, {query_id}!")
                        return index_info
        
        logging.debug(f"Indexing: {self.index_path}...")
        indexer = SHTIndexer(
            config=SHTIndexerConfig(
                use_hierarchy=self.embed_hierarchy,
                distance_metric=self.distance_metric,
                query_embedding_model_name=self.query_embedding_model,
            )
        )

        sht = self.build_sht(name)

        index_info = {
            "id": query_id,
            "indexes": indexer.index(query=query, nodes=sht["nodes"])
        }

        _write_to_file(
            dest_path=self.index_path,
            contents=json.dumps(index_info) + "\n",
            is_json=False,
            is_append=True
        )

        return index_info
    

    def generate_context(self, name, query, query_id):
        if os.path.exists(self.context_path):
            with open(self.context_path, 'r') as file:
                for l in file:
                    context_info = json.loads(l)
                    if int(context_info["id"]) == int(query_id):
                        logging.debug(f"Context already existed: {self.context_path}, {query_id}!")
                        return context_info
        
        dataset = None
        for ds in ["civic", "contract", "qasper", "finance"]:
            if ds in self.pdf_dir:
                assert dataset == None
                dataset = ds
        assert dataset != None
        true_context_len = get_context_len(
            context_ratio=self.context_len,
            # dataset=os.path.basename(os.path.dirname(self.pdf_dir)),
            dataset = dataset,
            sht_json_filename=name,
            min_context_len=round(max(self.chunk_size, self.summary_len) * 1.5),
        )
        logging.debug(f"Generating context: {self.context_path} (len={true_context_len})...")

        generator = SHTGenerator(
            config=SHTGeneratorConfig(
                use_hierarchy=self.context_hierarchy,
                use_raw_chunks=self.context_raw,
                context_len=true_context_len,
            )
        )

        index_info = self.index(name, query, query_id)

        sht = self.build_sht(name)

        # print(index_info["indexes"][:10])#########################delete this
        # print(name, self.sht_dir)
        # print("sht nodes # = ", len(sht["nodes"]))
        # for i in index_info["indexes"]:
        #     print(i)
        #     assert sht['nodes'][i['node_id']]['is_dummy'] == False

        context = generator.generate(
            candid_indexes=index_info["indexes"],
            nodes=sht["nodes"]
        )

        # This is for qasper dataset
        if os.path.basename(os.path.dirname(self.pdf_dir)) == "qasper":
            if name in context:
                context = context.replace(name, "")
            if "arXiv" in context:
                context = context.replace("arXiv", "")
            if "arxiv" in context:
                context = context.replace("arxiv", "")

        context_info = {
            "id": query_id,
            "context": context
        }
        print(f"Appending context (len={len(context)}) {self.context_path}...")
        _write_to_file(
            dest_path=self.context_path,
            contents=json.dumps(context_info) + "\n",
            is_json=False,
            is_append=True
        )

        return context_info
