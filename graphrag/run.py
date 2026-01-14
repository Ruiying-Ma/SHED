#Reference: https://github.com/run-llama/llama_index/blob/main/docs/examples/examples/cookbooks/GraphRAG_v1.ipynb

# import pandas as pd
import fsspec
import json
import os
import json
import asyncio
import logging
import logging_config
import nest_asyncio
import networkx as nx
from utils import get_document_txt
from typing import Any, List, Callable, Optional, Union, Dict
import tiktoken
import re
import argparse
from dotenv import load_dotenv
from graspologic.partition import hierarchical_leiden, HierarchicalClusters
# from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Document, StorageContext, load_index_from_storage, PropertyGraphIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import  EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.graph_stores.simple import SimpleGraphStore
# from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
# from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
# from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

# from llama_index.core import PropertyGraphIndex
# from llama_index.core.bridge.pydantic import BaseModel, Field
# from llama_index.legacy import StorageContext, load_index_from_storage
# from IPython.display import Markdown, display
nest_asyncio.apply()

CHUNK_SIZE = 100 # tokens, yl set 1024
CHUNK_OVERLAP = 0 # tokens, yl set 20
MAX_CLUSTER_SIZE = 5 # yl set 5
DEFAULT_PG_PERSIST_FNAME = "property_graph_store.json"


entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as (\"entity\"$$$$\"<entity_name>\"$$$$\"<entity_type>\"$$$$\"<entity_description>\")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as (\"relationship\"$$$$\"<source_entity>\"$$$$\"<target_entity>\"$$$$\"<relation>\"$$$$\"<relationship_description>\")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""



class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        logging.info(f"Instantiating GraphRAGExtractor...")

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata["entity_description"] = description  # Not used in the current implementation. But will be useful in future work.
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=metadata
            )
            existing_nodes.append(entity_node)

        metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, rel, obj, description = triple
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )

class GraphRAGStore(SimplePropertyGraphStore):
    # community_summary = {}
    max_cluster_size = 5

    def __init__(self, graph=None, community_summary={}):
        super().__init__(graph)
        self.community_summary = community_summary

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the graph store to a file."""
        if fs is None:
            fs = fsspec.filesystem("file")

        data = json.loads(self.graph.model_dump_json())
        assert "community_summary" not in data
        data["community_summary"] = self.community_summary
        
        with fs.open(persist_path, "w") as f:
            f.write(json.dumps(data))

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ):
        """Load from persist path."""
        if fs is None:
            fs = fsspec.filesystem("file")

        with fs.open(persist_path, "r") as f:
            data = json.loads(f.read())

        return cls.from_dict(data)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "SimplePropertyGraphStore":
        """Load from persist dir."""
        
        persist_path = os.path.join(persist_dir, DEFAULT_PG_PERSIST_FNAME)
        return cls.from_persist_path(persist_path, fs=fs)

    @classmethod
    def from_dict(cls, data: dict):
        # Use parent method to build the graph itself
        graph_store = SimplePropertyGraphStore.from_dict(data)
        # Restore community_summary if it exists in the dict
        assert "community_summary" in data
        community_summary = data.get("community_summary", {})
        # Wrap it in our subclass
        return cls(graph_store.graph, community_summary=community_summary)

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]

        openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        assert os.path.exists(openai_key_path)
        load_dotenv(openai_key_path)
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_API_KEY"),
            engine=os.getenv("AZURE_DEPLOYMENT_NAME")
        )
        response = client.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph: nx.Graph, clusters: HierarchicalClusters):
        """Collect detailed information for each node based on their community."""
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response


def store_index(index: PropertyGraphIndex, dataset, filename):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset, filename)
    os.makedirs(root_dir, exist_ok=False)
    index.storage_context.persist(
        persist_dir=root_dir,
    )

def load_index(dataset, filename):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset, filename)
    assert os.path.exists(root_dir)
    openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)

    storage_context = StorageContext.from_defaults(persist_dir=root_dir)
    property_graph_store = GraphRAGStore.from_persist_dir(persist_dir=root_dir)

    # print(property_graph_store.community_summary) ##################3

    # load index
    index: PropertyGraphIndex = load_index_from_storage(
        storage_context=storage_context,
        property_graph_store=property_graph_store,
    ) # PropertyGraphIndex
    # print(type(index))##############################
    # print(type(index.property_graph_store))###################
    return index

def graph_rag_build_graph(dataset, filename):
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset, filename)
    if os.path.exists(root_dir):
        logging.info(f"⚠️ Index already existed: {root_dir}!")
        return load_index(dataset, filename)
    
    assert not os.path.exists(root_dir)
    # llm = OpenAI(model=model)
    openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    llm = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
        engine=os.getenv("AZURE_DEPLOYMENT_NAME"),
    )

    # Create Document instance with the loaded text content
    logging.info(f"Loading text for {filename}...")
    doc_txt = get_document_txt(dataset, filename)
    documents = [Document(text=doc_txt)]

    # Split document into nodes
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # Initialize GraphRAGExtractor
    kg_extractor = GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )

    # Initialize PropertyGraphIndex and GraphRAGStore
    pg_store = GraphRAGStore()
    
    logging.info(f"Building PropertyGraphIndex...")
    index = PropertyGraphIndex(
        nodes=nodes,
        property_graph_store=pg_store,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )
    # Build communities
    assert isinstance(index.property_graph_store, GraphRAGStore)
    logging.info(f"Building communities...")
    index.property_graph_store.build_communities()

    store_index(index, dataset, filename)

    print(f"✅ {dataset}/{filename}: GraphRAG index is built and stored!")

    return index 


def graph_rag_query(index: PropertyGraphIndex, question):
    # llm = OpenAI(model=model)
    openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    llm = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
        engine=os.getenv("AZURE_DEPLOYMENT_NAME")
    )

    # Initialize Query Engine
    logging.info(f"Instantiating GraphRAGQueryEngine...")
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store, 
        llm=llm
    )

    # Query the engine
    response = query_engine.query(question)
    return response.response


# GOAL is to extract the table and key-value pairs from the PDFs
if __name__ == "__main__":

    ################################################# BUILD GRAPH #################################################
    # logging.disable(level=logging.DEBUG)
    # from datetime import datetime

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--terminal_id", type=int, required=True, help="Terminal ID")

    # DOC_NUM = 6

    # args = parser.parse_args()
    # start_id = (args.terminal_id - 1) * DOC_NUM

    # print(f"Processing docs [{start_id}, {start_id + DOC_NUM})...")
    
    # cur_doc_id = 0

    # for dataset in ["finance"]:

    #     pdf_folder = f"/home/ruiying/SHTRAG/data/{dataset}/pdf"
    #     assert os.path.exists(pdf_folder)

    #     for pdf_file in sorted(os.listdir(pdf_folder)):
    #         if cur_doc_id < start_id:
    #             cur_doc_id += 1
    #             continue
    #         if cur_doc_id >= start_id + DOC_NUM:
    #             exit(0)
    #         cur_doc_id += 1

    #         filename = pdf_file.replace(".pdf", "")
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"[{cur_time}] Processing doc: {dataset}/{filename}")
    #         try:
    #             graph = graph_rag_build_graph(dataset, filename)
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print(f"\t[{cur_time}]✅ {dataset}/{filename}: graph is built!")
    #         except Exception:
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print(f"\t[{cur_time}]❌ {dataset}/{filename}: failed to build graph!")
    ################################################ END BUILD GRAPH #################################################

    ################################################# QUERY GRAPH #################################################
    logging.disable(level=logging.DEBUG)
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal_id", type=int, required=True, help="Terminal ID")

    QUERY_NUM = 15

    args = parser.parse_args()
    start_id = (args.terminal_id - 1) * QUERY_NUM

    print(f"Processing queries [{start_id}, {start_id + QUERY_NUM})...")
    
    cur_query_id = 0
    for dataset in ["finance"]:
        queries_jsonl_path = f"/home/ruiying/SHTRAG/data/{dataset}/queries.json"
        assert os.path.exists(queries_jsonl_path)
        with open(queries_jsonl_path, "r") as f:
            queries = json.load(f)
        assert [q["id"] for q in queries] == list(range(len(queries)))

        answer_jsonl_path = f"/home/ruiying/SHTRAG/graphrag/{dataset}/answer_orig.jsonl"
        assert os.path.exists(os.path.dirname(answer_jsonl_path))
        answer_list = []
        if os.path.exists(answer_jsonl_path):
            with open(answer_jsonl_path, "r") as f:
                answer_list = [json.loads(line) for line in f.readlines() if len(line.strip()) > 0]

        for query_info in queries:
            if cur_query_id < start_id:
                cur_query_id += 1
                continue
            if cur_query_id >= start_id + QUERY_NUM:
                exit(0)
            cur_query_id += 1

            query_id = query_info["id"]
            filename = query_info["file_name"]
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{cur_time}] Processing query {query_id} from {dataset}/{filename}...")
            
            if query_id in [a["id"] for a in answer_list]:
                print(f"\t⚠️ Skipping existing query {query_id}...")
                continue

            query = query_info["prompt_template"]

            ### format query
            deleted_str = ""
            if dataset == "civic":
                deleted_str = "\n\n[Begin of Context]\n{context}\n[End of Context]"
            elif dataset == "contract":
                deleted_str = "\n\n[Begin of Contract Excerpt]\n{context}\n[End of Contract Excerpt]"
            elif dataset == "qasper":
                deleted_str = "\n\n[Begin of Context]\n{context}\n[End of Context]"
            elif dataset == "finance":
                deleted_str = "\n\n[Begin of Context]\n{context}\n[End of Context]"
            assert deleted_str in query
            query = query.replace(deleted_str, "")

            graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset, filename)
            if not os.path.exists(graph_dir):
                print(f"\t❗ Skipping query {query_id} for non-existing graph {dataset}/{filename}...")
                continue

            graph = graph_rag_build_graph(dataset, filename)
            try:
                response = graph_rag_query(graph, query)
                answer = {
                    "id": query_id,
                    "answer": response
                }
                with open(answer_jsonl_path, "a") as f:
                    f.write(json.dumps(answer) + "\n")
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]✅ Written answer {query_id} to {answer_jsonl_path}!")
            except Exception as e:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]❌ Answer {query_id} failed!\n\t\t{e}")

    ####################################################

