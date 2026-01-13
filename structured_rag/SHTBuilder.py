import json
import os
import sys
from typing import Dict, List
import tiktoken
from anytree import AnyNode, RenderTree

from .EmbeddingModels import SBertEmbeddingModel, DPRContextEmbeddingModel, TextEmbedding3SmallModel
from .SummarizationModels import GPT4oMiniSummarizationModel, EmptySummarizationModel
from .utils import get_nondummy_ancestors, split_text_into_sentences, split_text_into_chunks

import logging

class SHTBuilderConfig:
    def __init__(self, store_json: str, load_json: str, chunk_size: int, summary_len: int, embedding_model_name: str, summarization_model_name: str, openai_key_path: str=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")):
        if isinstance(store_json, str):
            if not store_json.endswith(".json"):
                raise ValueError(f"store_json ({store_json}) must be a path to a json file that ends with .json")
            elif not os.path.exists(os.path.dirname(store_json)):
                raise ValueError(f"store_json directory ({store_json}) doesn't exist")
        else:
            raise ValueError(f"store_json ({store_json}) must be a valid string")

        if load_json is not None:
            if not load_json.endswith(".json"):
                raise ValueError(f"load_json ({load_json}) must be a path to a json file that ends with .json")
            elif not os.path.exists(os.path.dirname(load_json)):
                raise ValueError(f"load_json directory ({load_json}) doesn't exist")

        if (not isinstance(chunk_size, int)) or (chunk_size <= 0):
            raise ValueError(f"chunk_size must be a positive integer, but you give {chunk_size}")

        if (not isinstance(summary_len, int)) or (summary_len <= 0):
            raise ValueError(f"summary_len must be a positive integer, but you give {summary_len}")

        # if (not isinstance(embedding_model_name, str)) or (not embedding_model_name in ["sbert", "dpr", "te3small"]):
        #     raise ValueError(f"embedding_model_name must be 'sbert', 'dpr', or 'te3small', but you give {embedding_model_name}")

        # if (not isinstance(openai_key_path, str)) or (not os.path.exists(openai_key_path)):
        #     raise ValueError(f"openai_key_path should be a path to the .env file storing openai key, but you give {openai_key_path}")
        
        self.chunk_size = chunk_size
        self.store_json = store_json
        self.load_json = load_json
        self.summary_len = summary_len
        self.embedding_model_name = embedding_model_name
        self.summarization_model_name = summarization_model_name
        self.openai_key_path = openai_key_path
        

class SHTBuilder:
    def __init__(self, config: SHTBuilderConfig):
        if not isinstance(config, SHTBuilderConfig):
            raise ValueError("config should be an instance of SHTBuilderConfig")
        
        self.chunk_size = config.chunk_size
        self.store_json = config.store_json
        self.load_json = config.load_json
        self.summary_len = config.summary_len
        self.embedder = None
        if config.embedding_model_name == "sbert":
            self.embedder = SBertEmbeddingModel()
        elif config.embedding_model_name == "dpr":
            self.embedder = DPRContextEmbeddingModel()
        elif config.embedding_model_name == "te3small":
            self.embedder = TextEmbedding3SmallModel(openai_key_path=config.openai_key_path)
        else:
            raise ValueError(f"Unknown embedding_model_name {config.embedding_model_name}")
        assert self.embedder is not None
        self.summarizer = None
        if config.summarization_model_name == "gpt-4o-mini":
            self.summarizer = GPT4oMiniSummarizationModel(openai_key_path=config.openai_key_path)
        elif config.summarization_model_name == "empty":
            self.summarizer = EmptySummarizationModel()
        else:
            raise ValueError(f"Unknown summarization_model_name {config.summarization_model_name}")
        # Default values
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tree = None

    def reset(self):
        '''
        Clear the tree
        '''
        self.tree = None

    def _create_node(self, id: int, type: str, parent: int, heading: str, texts: List[str], info: Dict) -> Dict:
        '''
        Create a non-dummy node. 
        
        A non-dummy node is a dictionary:
        ```
        {
            "is_dummy": bool, # False
            "id": int, # 0-based,
            "type": str, # "head" (Section headers/Titles), "list" (List items), or "text" (newly added leaves, i.e., chunks)
            "parent": int, # id of its parent. If no parent, -1.
            "children": List[int], # list of ids of its children in ascending order
            "nondummy_parent": int, # id of its non-dummy parent
            "nondummy_children": List[int], # ids of its non-dummy children in ascending order
            "heading": str # the heading attribute; for list-type nodes, this is the first sentence; for newly added leaves (text-type), this is empty.
            "texts": List[str], # the context attribute; for head-/list-type nodes, this is recursively generated by summarization, stored in `texts[0]`; for newly added leaves (text-type), this is the list of the corresponding chunks.
            "embeddings": Dict[List], # the embedding attribute; "heading": an embedding for heading attribute if exists (only head-/list-type nodes have); "texts": an embedding for context attribute; "hybrid": the embedding including contextual and hierarchical information, as stated in the paper
            "info": Dict, # the visual patterns and the cluster returned by `ClusteringOracle`. This is `None` or text-type nodes.
        }
        ```
        Initially, children and embeddings are not created. Children will be created after the parental relations are determined. Embeddings will be created after the whole SHT are created. Also, initially, "texts" for head-/list-type nodes are [""]. They will be created after the structure of the SHT are determined. 
        '''
        if not isinstance(id, int):
            raise ValueError("id must be an integer")

        if not isinstance(type, str) or not type in ["head", "list", "text"]:
            raise ValueError("type must be one of 'head', 'list', 'text'")

        if not isinstance(parent, int):
            raise ValueError("parent must be an integer")

        if not isinstance(heading, str):
            raise ValueError("heading must be a string")
        elif heading == "":
            if type != "text":
                raise ValueError("only text-type nodes can have empty heading")
        else:
            if type == "text":
                raise ValueError("text-type nodes must have empty heading")

        if not isinstance(texts, List):
            raise ValueError("texts must be a list")
        elif len(texts) > 0:
            if not all([isinstance(t, str) for t in texts]):
                raise ValueError("texts must be a list of string")
            if type != "text" and len(texts) > 1:
                raise ValueError("head-/list-type nodes cannot have more than 1 item in texts")
        else:
            if type == "text":
                raise ValueError("text-type nodes' texts must be non-empty")

        node = {
            "is_dummy": False,
            "id": id,
            "type": type,
            "parent": parent,
            "children": [],
            "heading": heading,
            "texts": texts,
        }

        if type == "head" or type == "list":
            if info is None:
                raise ValueError("head-/list-type nodes' info cannot be None")

            node["embeddings"] = {
                "heading": None,
                "texts": None,
                "hybrid": None,
            }
            node["info"] = info
        else:
            assert type == "text"
            if info is not None:
                raise ValueError("text-type nodes' info must be None")
            node["embeddings"] = {
                "texts": None,
                "hybrid": None,
            }

        return node

    def _create_dummy(self, id: int, parent: int) -> Dict:
        '''
        Create a dummy node.

        A dummy node is a dictionary:
        ```
        {
            "is_dummy": bool, # True
            "id": int,
            "parent": int,
            "children": List[int],
            "non_dummy": int,
            "nondummy_children": List[int]
        }
        ```

        Initially children are not given. Children will be created after parent relations are determined.
        '''

        if not isinstance(id, int):
            raise ValueError("id must be an integer")

        if not isinstance(parent, int):
            raise ValueError("parent must be an integer")

        return {
            "is_dummy": True,
            "id": id,
            "parent": parent,
            "children": [],
        }

    def _update_parent(self, node_id, parent_id, nodes):
        '''
        Update the "children" list of parent node.
        '''
        if parent_id == -1:
            return
        assert parent_id < len(nodes)
        parent_node = nodes[parent_id]
        assert parent_node["id"] == parent_id
        children = parent_node["children"]
        assert node_id not in children
        children.append(node_id)
        assert all([c <= node_id for c in children])

    def _place_node(self, pre_node: Dict, rpath: List[int], m_cluster_to_height: Dict, nodes: List, mode=None) -> int:
        '''
        SHTgen. Determine where to place a node. 

        Args:
        - pre_node: the node to be placed into current SHT
        - rpath: the rightmost path (the ids of the nodes in ascending order)
        - m_cluster_to_height: record the cluster to its level (height) in the SHT
        - nodes: current SHT

        Returns:
        - the id of the placed node
        '''
        assert pre_node["type"] in ["head", "list"]
        if mode == None and len(rpath) > 1:
            assert set([nodes[i]["type"] for i in rpath[1:] if not nodes[i]["is_dummy"]]) == set([pre_node["type"]])
        assert len(rpath) > 0
        assert sorted(rpath) == rpath

        pre_node_cluster = pre_node["info"]["cluster_id"]
        if pre_node_cluster not in m_cluster_to_height:
            # new cluster, append node to rpath
            node = self._create_node(
                id=len(nodes),
                type=pre_node["type"],
                parent=rpath[-1],
                heading=pre_node["heading"],
                texts=[""],
                info=pre_node["info"],
            )

            m_cluster_to_height[pre_node_cluster] = len(rpath) - 1
            rpath.append(node["id"])
            nodes.append(node)
        else:
            height = m_cluster_to_height[pre_node_cluster]
            while len(rpath) < height + 1:
                dummy_node = self._create_dummy(
                    id=len(nodes),
                    parent=rpath[-1],
                )

                rpath.append(dummy_node["id"])
                nodes.append(dummy_node)
                self._update_parent(node_id=dummy_node["id"], parent_id=dummy_node["parent"], nodes=nodes)

            node = self._create_node(
                id=len(nodes),
                type=pre_node["type"],
                parent=rpath[height],
                heading=pre_node["heading"],
                texts=[""],
                info=pre_node["info"],
            )
            
            # new_rpath = deepcopy([id for id in rpath if id <= node["parent"]])
            rpath[:] = rpath[:(height+1)]
            rpath.append(node["id"])
            nodes.append(node)

        self._update_parent(node_id=node["id"], parent_id=node["parent"], nodes=nodes)

        return node["id"]

    def build(self, object_dicts_list: List[Dict], mode=None):
        '''
        SHTgen. Build the SHT.

        Args:
            - object_dicts_list (List[Dict]): the objects returned by `ClusteringOracle`
            - mode: None, wide, deep

        Store the sht (Dict):
            - stored in self.tree
            - nodes (List[Dict]): the list of SHT nodes in preorder
            - m_height_to_ids_list (Dict): a map from node height to node id (0-based). -1 is the height of the fake root.
            - m_id_to_height (Dict): a map from node id to node height. -1 is the id of the fake root.
            - full_text (str): joined by \\n\\n
            - estimated_cost (Dict): input_tokens, output_tokens
        '''
        assert mode in [None, 'wide', 'deep']
        
        # preprocess the objects, such that only three types of object exists: "text", "head", "list"
        self.reset()

        if self.load_json is not None:
            logging.info(f"Loading existed SHT from {self.load_json}")
            with open(self.load_json, 'r') as file:
                self.tree = json.load(file)
            self.check()
            return
        
        pre_nodes_list = []
        for object_dict in object_dicts_list:
            object_type = object_dict["type"]
            assert object_dict["text"] != ""
            if object_type == "Section header" or object_type == "Title":
                pre_node = {
                    "id": len(pre_nodes_list),
                    "type": "head",
                    "heading": object_dict["text"],
                    "info": {
                        "features": object_dict["features"],
                        "cluster_id": object_dict["cluster_id"]
                    }
                }
                pre_nodes_list.append(pre_node)
            elif object_type == "List item":
                full_text = object_dict["text"]
                subsent_delimiters = [".", "!", "?", "\n", ",", ";", ":"]
                subsentences = split_text_into_sentences(subsent_delimiters, full_text)
                heading = subsentences[0]
                prev_subsentence_id = 0
                while prev_subsentence_id + 1 < len(subsentences) and len(self.tokenizer.encode(heading + subsentences[prev_subsentence_id + 1])) < 20:
                    heading += subsentences[prev_subsentence_id + 1]
                    prev_subsentence_id += 1
                assert subsentences[prev_subsentence_id] in full_text and subsentences[prev_subsentence_id] in heading
                pre_node = {
                    "id": len(pre_nodes_list),
                    "type": "list",
                    "heading": heading,
                    "info": {
                        "features": object_dict["features"],
                        "cluster_id": object_dict["cluster_id"]
                    }
                }
                pre_nodes_list.append(pre_node)
                # create a new text pre_node
                remain_text_starter = full_text.find(subsentences[prev_subsentence_id])
                assert remain_text_starter != -1
                remain_text = full_text[remain_text_starter + len(subsentences[prev_subsentence_id]):]
                if remain_text.strip() != "":
                    new_pre_node = {
                        "id": len(pre_nodes_list),
                        "type": "text",
                        "text": remain_text,
                    }
                    pre_nodes_list.append(new_pre_node)
            else:
                assert object_type in ["Caption", "Footnote", "Formula", "Page footer", "Page header", "Table", "Text"]
                if len(pre_nodes_list) == 0 or pre_nodes_list[-1]["type"] != "text":
                    # create a pre_node
                    pre_node = {
                        "id": len(pre_nodes_list),
                        "type": "text",
                        "text": object_dict["text"]
                    }
                    pre_nodes_list.append(pre_node)
                else:
                    last_pre_node = pre_nodes_list[-1]
                    if last_pre_node["type"] == "text":
                        # directly add text using "\n\n" as joiner
                        last_pre_node["text"] += "\n\n" + object_dict["text"]
        
        # check pre_nodes_list
        assert [n["id"] for n in pre_nodes_list] == list(range(len(pre_nodes_list)))
        for pre_node in pre_nodes_list:
            if pre_node["type"] == "text":
                id = pre_node["id"]
                if id - 1 >= 0:
                    assert pre_nodes_list[id - 1]["type"] != "text"
                if id + 1 < len(pre_nodes_list):
                    assert pre_nodes_list[id + 1]["type"] != "text"
        
        # build the initial tree
        nodes = []
        m_cluster_to_height_global = dict()
        m_cluster_to_height_local = dict()
        prev_head_or_list_id = -1
        rpath_global = [-1] # list of ids of the rpath starting from height=0
        rpath_local = [-1]
        for pre_node in pre_nodes_list:
            pre_node_type = pre_node["type"]
            assert pre_node_type in ["head", "text", "list"]
            if pre_node_type == "head":
                # place the node, update global info
                new_node_id = self._place_node(
                    pre_node=pre_node,
                    rpath=rpath_global,
                    m_cluster_to_height=m_cluster_to_height_global,
                    nodes=nodes,
                    mode=mode,
                )
                prev_head_or_list_id = new_node_id
                # reset local info
                m_cluster_to_height_local = dict()
                rpath_local = [new_node_id]
            elif pre_node_type == "list":
                # place the node, update local info
                if mode == None:
                    new_node_id = self._place_node(
                        pre_node=pre_node,
                        rpath=rpath_local,
                        m_cluster_to_height=m_cluster_to_height_local,
                        nodes=nodes,
                        mode=mode,
                    )
                else:
                    new_node_id = self._place_node(
                        pre_node=pre_node,
                        rpath=rpath_global,
                        m_cluster_to_height=m_cluster_to_height_global,
                        nodes=nodes,
                        mode=mode
                    )
                prev_head_or_list_id = new_node_id
            else:
                assert pre_node_type == "text"
                # place the node
                new_node = self._create_node(
                    id=len(nodes),
                    type="text",
                    parent=prev_head_or_list_id,
                    heading="",
                    texts=split_text_into_chunks(self.chunk_size, pre_node["text"], self.tokenizer),
                    info=None,
                )
                nodes.append(new_node)
                self._update_parent(node_id=new_node["id"], parent_id=prev_head_or_list_id, nodes=nodes)

        # self._check_tree_building(nodes)

        # get tree meta data
        m_height_to_ids_list = dict()
        m_id_to_height = dict()
        full_text = ""

        m_id_to_height[-1] = -1
        m_height_to_ids_list[-1] = [-1]
        for node in nodes:
            height = m_id_to_height[node["parent"]] + 1
            m_id_to_height[node["id"]] = height
            if height not in m_height_to_ids_list:
                m_height_to_ids_list[height] = list()
            m_height_to_ids_list[height].append(node["id"])
            if not node["is_dummy"]:
                if node["type"] == "head" or node["type"] == "list":
                    full_text += node["heading"]
                    full_text += "\n\n"
                else:
                    full_text += "\n\n".join(node["texts"])
                    if full_text != "":
                        full_text += "\n\n"

        assert self.tree is None
        assert all([sorted(l) == l for l in m_height_to_ids_list.values()])
        self.tree = {
            "nodes": nodes,
            "m_height_to_ids_list": {str(k): v for k, v in m_height_to_ids_list.items()},
            "m_id_to_height": {str(k): v for k, v in m_id_to_height.items()},
            "full_text": full_text,
        }

        self._estimate_cost()
        self._add_nondummy_parent()
        self._add_nondummy_children()
        self.check()

    def _check_tree_building(self, nodes: List[Dict]):
        '''
        Check the initial tree built by self.build()
        '''
        # check id
        assert list([n["id"] for n in nodes]) == list(range(len(nodes)))
        # check preorder
        root = AnyNode(text="ROOT", id=-1)
        vis_node = {
            -1: root
        }
        for node in nodes:
            assert node["parent"] in vis_node
            if node["is_dummy"]:
                text = "DUMMY"
            else:
                text = f'''({node["type"]}) '''
                if node["type"] == "text":
                    text += f'''{node["texts"][0][:min(10, len(node["texts"][0]))]} '''
                    text += "......"
                    text += f''' {node["texts"][-1][-min(10, len(node["texts"][-1])):]}'''
                else:
                    text += node["heading"]
            vis_node[node["id"]] = AnyNode(text=text, id=node["id"], parent=vis_node[node["parent"]])
        id = -1
        for pre, _, node in RenderTree(node=root, maxlevel=900):
            assert node.id == id
            id += 1
        # check children
        assert all([n["children"] == sorted(n["children"]) for n in nodes])
        assert all([len(n["children"]) == 0 for n in nodes if (not n["is_dummy"]) and (n["type"] == "text")])
        assert all([len(n["children"]) > 0 for n in nodes if n["is_dummy"]])
        # check structure (child-parent relation)
        for node in nodes:
            
            parent_id = node["parent"]
            assert parent_id >= -1 and parent_id < len(nodes)
            if parent_id != -1:
                assert node["id"] in nodes[parent_id]["children"]
            for child_id in node["children"]:
                assert child_id >= 0 and child_id < len(nodes)
                assert nodes[child_id]["parent"] == node["id"]
        # check contents
        for node in nodes:
            if node["is_dummy"]:
                continue
            if node["type"] == "text":
                assert node["heading"] == ""
                assert len(node["texts"]) > 0
            else:
                assert node["heading"] != ""
                assert len(node["texts"]) == 1
        
    def _estimate_cost(self):
        '''
        Estimate the LLM cost of SHT construction. Each non-dummy node output its heading + texts once. For input_tokens, take the sum of nodes that have parents. For input_tokens, take the sum of nodes that have children. 

        Args:
            - tree (Dict): sht

        Add to self.tree the "estimated_cost" (Dict):
            - input_tokens (int)
            - output_tokens (int)
        '''
        if self.tree is None:
            raise ValueError("An intial tree must be built before estimating the cost of building its summaries")
        
        input_tokens = 0
        output_tokens = 0
        for node in self.tree["nodes"]:
            if node["is_dummy"]:
                continue
            if node["type"] == "text":
                tokens = len(self.tokenizer.encode("\n\n".join(node["texts"]) + "\n\n"))
            else:
                tokens = len(self.tokenizer.encode(node["heading"] + "\n\n")) + self.summary_len
            
            if node["parent"] != -1:
                input_tokens += tokens
            if len(node["children"]) > 0:
                output_tokens += tokens
            
        estimated_cost = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        self.tree["estimated_cost"] = estimated_cost
            
    def _add_nondummy_parent(self):
        '''
        After building the initial tree, before adding summary, add immediate nondummy parents to all nodes. nondummy_parent is added to each self.tree["nodes"] for debugging purpose.
        '''
        def _get_nondummy_parent(node_id: int) -> int:
            '''
            Get the immediate nondummy parent of node
            '''
            if self.tree is None:
                raise ValueError("self.tree doesn't exist")

            if not isinstance(node_id, int) or node_id >= len(self.tree["nodes"]) or node_id < 0:
                raise ValueError(f"node {node_id} doesn't exist")

            node = self.tree["nodes"][node_id]
            if node["parent"] == -1:
                return -1
            if "nondummy_parent" in node:
                return node["nondummy_parent"]
            parent_node = self.tree["nodes"][node["parent"]]
            if parent_node["is_dummy"]:
                return _get_nondummy_parent(parent_node["id"])
            else:
                return parent_node["id"]
            
        for node in self.tree["nodes"]:
            node["nondummy_parent"] = _get_nondummy_parent(node["id"])
    
    def _add_nondummy_children(self):
        '''
        After building the initial tree, before adding summary, add immediate nondummy children to all nodes. nondummy_children is added to each self.tree["nodes"] for debugging purpose.
        '''
        def _get_nondummy_children(node_id: int) -> List[int]:
            '''
            Get the immediate nondummy children of node
            '''
            if self.tree is None:
                raise ValueError("self.tree doesn't exist")

            if not isinstance(node_id, int) or node_id >= len(self.tree["nodes"]) or node_id < 0:
                raise ValueError(f"node {node_id} doesn't exist")

            node = self.tree["nodes"][node_id]
            if len(node["children"]) == 0:
                return []

            if "nondummy_children" in node:
                return node["nondummy_children"]
            
            nondummy_children = []
            for child_id in node["children"]:
                child_node = self.tree["nodes"][child_id]
                if not child_node["is_dummy"]:
                    nondummy_children.append(child_id)
                else:
                    nondummy_children += _get_nondummy_children(child_id)

            
            assert len(nondummy_children) >= len(node["children"])
            assert len(nondummy_children) > 0
            assert len(node["children"]) > 0
            assert all([nid > node_id for nid in nondummy_children])
            assert sorted(nondummy_children) == nondummy_children

            return nondummy_children

        for node in self.tree["nodes"]:
            node["nondummy_children"] = _get_nondummy_children(node["id"])
    
    def _check_nondummy_relations(self):
        '''
        After adding nondummy parents and children, check
        '''
        for node in self.tree["nodes"]:
            assert "nondummy_parent" in node and "nondummy_children" in node
            if node["nondummy_parent"] != -1:
                assert not self.tree["nodes"][node["nondummy_parent"]]["is_dummy"]
                if not node["is_dummy"]:
                    assert node["id"] in self.tree["nodes"][node["nondummy_parent"]]["nondummy_children"]
            
            for nondummy_child_id in node["nondummy_children"]:
                assert not self.tree["nodes"][nondummy_child_id]["is_dummy"]
                if not node["is_dummy"]:
                    assert self.tree["nodes"][nondummy_child_id]["nondummy_parent"] == node["id"]

    def _check_node(self, node: Dict):
        '''
        Check node's structure
        '''
        if not isinstance(node, Dict):
            raise ValueError("node should be of type Dict")

        if not "is_dummy" in node or not isinstance(node["is_dummy"], bool):
            raise ValueError("node should have is_dummy")

        if not "id" in node or not isinstance(node["id"], int) or node["id"] < 0:
            raise ValueError("node should have nonnegative id")

        if not "parent" in node or not isinstance(node["parent"], int) or node["parent"] < -1:
            raise ValueError("node should have parent")

        if not "children" in node or not isinstance(node["children"], List) or not all([(isinstance(c, int) and c >= 0) for c in node["children"]]):
            raise ValueError("node should have children")

        if not "nondummy_parent" in node or not isinstance(node["nondummy_parent"], int) or node["nondummy_parent"] < -1:
            raise ValueError("node should have nondummy_parent")

        if not "nondummy_children" in node or not isinstance(node["nondummy_children"], List) or not all([(isinstance(c, int) and c >= 0) for c in node["children"]]):
            raise ValueError("node should have nondummy_children")

        if node["is_dummy"]:
            return True

        if not "type" in node or (not node["type"] in ["head", "list", "text"]):
            raise ValueError("nondummy node should have type")

        if not "heading" in node or not isinstance(node["heading"], str):
            raise ValueError("nondummy node should have heading")
        else:
            if node["type"] == "text" and node["heading"] != "":
                raise ValueError("text node should have empty heading")
            elif node["type"] != "text" and node["heading"] == "": 
                raise ValueError("head/list node should have nonempty heading")

        if not "texts" in node or not isinstance(node["texts"], List):
            raise ValueError("nondummy node should have texts")
        else:
            if node["type"] != "text":
                if len(node["texts"]) != 1:
                    raise ValueError("head/list node should have 1 text")
                elif not all([isinstance(t, str) for t in node["texts"]]):
                    raise ValueError("text in texts should be of type string")
            else:
                if len(node["texts"]) == 0 or not all([isinstance(t, str) for t in node["texts"]]):
                    raise ValueError("text node should have nonempty texts")

        if not "embeddings" in node or not isinstance(node["embeddings"], Dict):
            raise ValueError("nondummy node should have embeddings")
        else:
            if "texts" not in node["embeddings"] or "hybrid" not in node["embeddings"]:
                raise ValueError("nondummy node should have texts- and hybrid- embeddings")
            if node["type"] != "text":
                if "heading" not in node["embeddings"]:
                    raise ValueError("head/list node should have heading-embeddings")
            else:
                if "heading" in node["embeddings"]:
                    raise ValueError("text node should not have heading-embeddings")

        return True

    def check(self):
        '''
        Universal check
        '''
        if self.tree is None:
            raise ValueError("self.tree doesn't exist")
        if not "nodes" in self.tree or not isinstance(self.tree["nodes"], List) or not all([self._check_node(n) for n in self.tree["nodes"]]):
            raise ValueError("self.tree has invalid node")
        
        if not "m_height_to_ids_list" in self.tree or not isinstance(self.tree["m_height_to_ids_list"], Dict):
            raise ValueError("self.tree doesn't have m_height_to_ids_list")
        else:
            for height in self.tree["m_height_to_ids_list"]:
                assert all([isinstance(id, int) for id in self.tree["m_height_to_ids_list"][height]])
                assert sorted(self.tree["m_height_to_ids_list"][height]) == self.tree["m_height_to_ids_list"][height]
        
        if not "m_id_to_height" in self.tree or not isinstance(self.tree["m_id_to_height"], Dict):
            raise ValueError("self.tree doesn't have m_id_to_height")

        if not "full_text" in self.tree or not isinstance(self.tree["full_text"], str):
            raise ValueError("self.tree doesn't have full_text")

        if not "estimated_cost" in self.tree or not isinstance(self.tree["estimated_cost"], Dict) or (not ("input_tokens" in self.tree["estimated_cost"] and "output_tokens" in self.tree["estimated_cost"] and isinstance(self.tree["estimated_cost"]["input_tokens"], int) and isinstance(self.tree["estimated_cost"]["output_tokens"], int))):
            raise ValueError("self.tree doesn't have estimated_cost")

        self._check_tree_building(self.tree["nodes"])
        self._check_nondummy_relations()

    def add_summaries(self) -> Dict:
        '''
        Populate the context attribute "texts" for head-/list-type nodes by recursive summarization. The texts are concatenated by \\n\\n.

        Returns:
            - stats (Dict): statistics of summarizing the tree
                - input_tokens (int)
                - output_tokens (int)
                - time (float)
        '''
        
        logging.info(f"Adding summaries for {os.path.basename(self.store_json)}")

        self.check()
        
        heights = sorted([int(h) for h in list(self.tree['m_height_to_ids_list'].keys())])[::-1]
        assert heights[-1] == -1
        assert heights[::-1] == list(range(-1, len(heights) - 1))
        node_id_has_summary =set()
        stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "time": 0.0,
        }
        for h in heights:
            if h == -1: # root
                assert all([n["id"] in node_id_has_summary for n in self.tree["nodes"] if not n["is_dummy"]])
                return stats
            height = str(h)
            node_ids_list = self.tree["m_height_to_ids_list"][height]
            assert node_ids_list == sorted(node_ids_list)
            for node_id in node_ids_list:
                node = self.tree["nodes"][node_id]

                if node["is_dummy"]:
                    continue

                if len(node["nondummy_children"]) == 0:
                    node_id_has_summary.add(node["id"])
                    
                else:
                    assert node["type"] != "text"
                    assert all([cid in node_id_has_summary for cid in node["nondummy_children"]])
                    assert sorted(node["nondummy_children"]) == node["nondummy_children"]
                    assert node["id"] not in node_id_has_summary
                    text_for_summarization = ""
                    for child_id in node["nondummy_children"]:
                        cur_text = ""
                        child_node = self.tree["nodes"][child_id]
                        assert not child_node["is_dummy"]
                        cur_text += child_node["heading"]
                        if child_node["heading"] != "":
                            cur_text += "\n\n"
                        if child_node["type"] != "text":
                            assert len(child_node["texts"]) == 1
                        for chunk in child_node["texts"]:
                            cur_text += chunk
                            if chunk != "":
                                cur_text += "\n\n"
                        if cur_text != "":
                            text_for_summarization += cur_text
                    try:
                        summary_info = self.summarizer.summarize(text_for_summarization, self.summary_len)
                    except Exception:
                        true_sum_len = min(len(text_for_summarization), self.summary_len)
                        summary_info = {
                            "summary": text_for_summarization[:true_sum_len],
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "time": 0.0
                        }
                    # summary_info = {
                    #     "summary": "",
                    #     "input_tokens": 0,
                    #     "output_tokens": 0,
                    #     "time": 0.0,
                    # }
                    summary = summary_info["summary"]
                    node["texts"][0] = summary
                    stats["input_tokens"] += summary_info["input_tokens"]
                    stats["output_tokens"] += summary_info["output_tokens"]
                    stats["time"] += summary_info["time"]
                    node_id_has_summary.add(node["id"])
        
        raise ValueError("Executions error: must not reach to this line")

    def _get_nondummy_ancestors(self, node_id: int) -> List[int]:
        '''
        Return the list of nondummy ancestors' ids of the given nondummy node in ascending order. Note that -1 (the fake root) is not contained.
        '''
        if self.tree is None:
            raise ValueError("self.tree doesn't exist")

        if not isinstance(node_id, int) or node_id < 0 or node_id >= len(self.tree["nodes"]):
            raise ValueError("node doesn't exist")

        return get_nondummy_ancestors(self.tree["nodes"], node_id)
        
        # node = self.tree["nodes"][node_id]

        # if node["is_dummy"]:
        #     raise ValueError("node is dummy")

        # if "nondummy_parent" not in node:
        #     raise ValueError("Must execute self._add_nondummy_parent() in advance")

        # cur_parent_id = node["nondummy_parent"]
        # ancestors = []
        # while cur_parent_id != -1:
        #     ancestors.append(cur_parent_id)
        #     cur_node = self.tree["nodes"][cur_parent_id]
        #     assert not cur_node["is_dummy"]
        #     cur_parent_id = cur_node["nondummy_parent"]

        # assert ancestors[::-1] == sorted(ancestors)
        # if len(ancestors) > 0:
        #     assert ancestors[0] == node["nondummy_parent"]
        # return ancestors[::-1]

    def add_embeddings(self, node_ids_list: List[int]) -> float:
        '''
        Populate the embedding attributes after filling out the contexts attributes (i.e., the summaries).

        Returns:
            - the time cost:
                - hybrid
                - texts
                - heading
        '''
        logging.info(f"Adding embeddings for {os.path.basename(self.store_json)}")

        if self.tree is None:
            raise ValueError("tree doesn't exists")

        if not all([(id >= 0 and id < len(self.tree["nodes"])) for id in node_ids_list]):
            raise ValueError("Invalid node id")


        time_cost = {
            "hybrid": 0.0,
            "texts": 0.0,
            "heading": 0.0,
        }

        for node_id in node_ids_list:
            node = self.tree["nodes"][node_id]
            if node["is_dummy"]:
                continue
            
            ancestors = self._get_nondummy_ancestors(node_id)
            if len(ancestors) > 0:
                assert sorted(ancestors) == ancestors
                assert all([(aid >= 0 and aid < len(self.tree["nodes"]) for aid in ancestors)])
                assert all([not self.tree["nodes"][aid]["is_dummy"] for aid in ancestors])
                assert all([self.tree["nodes"][aid]["type"] != "text" for aid in ancestors])
                assert all([self.tree["nodes"][aid]["heading"] != "" for aid in ancestors])
                assert sorted(ancestors) == ancestors
            ancestor_string = "\n\n".join([self.tree["nodes"][aid]["heading"] for aid in ancestors])
            if ancestor_string != "":
                ancestor_string += "\n\n"

            heading_string = node["heading"]
            if heading_string != "":
                heading_string += "\n\n"

            if node["type"] == "text":
                assert set(node["embeddings"].keys()) == set(["texts", "hybrid"])
                assert node["heading"] == "" and heading_string == ""
            else:
                assert node["type"] in ["head", "list"]
                assert set(node["embeddings"].keys()) == set(["texts", "hybrid", "heading"])
                assert node["heading"] != ""
                heading_embedding_info = self.embedder.create_embedding(node["heading"])
                node["embeddings"]["heading"] = heading_embedding_info["embedding"]
                time_cost["heading"] += heading_embedding_info["time"]
            
            text_embeddings_info = [self.embedder.create_embedding(heading_string + t) for t in node["texts"]]
            node["embeddings"]["texts"] = [info["embedding"] for info in text_embeddings_info]
            time_cost["texts"] += sum([info["time"] for info in text_embeddings_info])

            hybrid_embeddings_info = [self.embedder.create_embedding(ancestor_string + heading_string + t) for t in node["texts"]]
            node["embeddings"]["hybrid"] = [info["embedding"] for info in hybrid_embeddings_info]
            time_cost["hybrid"] += sum([info["time"] for info in hybrid_embeddings_info])
        # check
        for node_id in node_ids_list:
            node = self.tree["nodes"][node_id]
            if node["is_dummy"]:
                continue
            for key in node["embeddings"]:
                assert node["embeddings"] is not None
                if key in ["texts", "hybrid"]:
                    assert len(node["embeddings"][key]) == len(node["texts"])

        return time_cost

    def visualize(self, vis_path):
        '''
        Visualize the tree
        '''
        if self.tree is None:
            raise ValueError("tree doesn't exist")

        nodes = self.tree["nodes"]
        root = AnyNode(text="ROOT", id=-1)
        vis_node = {
            -1: root
        }
        for node in nodes:
            assert node["parent"] in vis_node
            if node["is_dummy"]:
                text = "DUMMY"
            else:
                text = f'''({node["type"]}) '''
                if node["type"] == "text":
                    text += f'''{node["texts"][0][:min(10, len(node["texts"][0]))]} '''
                    text += "......"
                    text += f''' {node["texts"][-1][-min(10, len(node["texts"][-1])):]}'''
                else:
                    text += node["heading"]
            vis_node[node["id"]] = AnyNode(text=text, id=node["id"], parent=vis_node[node["parent"]])

        with open(vis_path, 'w') as file:
            vis_tree_str = ""
            for pre, _, node in RenderTree(node=root, maxlevel=900):
                vis_tree_str += f"{pre}{node.id}: {node.text}\n"
            file.write(vis_tree_str)

    def store2json(self):
        if self.tree is None:
            raise ValueError("self.tree doesn't exist")
        assert not os.path.exists(self.store_json)
        with open(self.store_json, 'w') as file:
            json.dump(self.tree, file, indent=4)

    def _get_nondummy_successor(self, node_id: int) -> int:
        if self.tree is None:
            raise ValueError("self.tree doesn't exist")

        if not isinstance(node_id, int) or node_id < 0 or node_id >= len(self.tree["nodes"]):
            raise ValueError("node doesn't exist")
        
        node = self.tree["nodes"][node_id]

        if node["is_dummy"]:
            raise ValueError("node is dummy")

        if "nondummy_parent" not in node:
            raise ValueError("Must execute self._add_nondummy_parent() in advance")

        for id in range(node["id"] + 1, len(self.tree["nodes"])):
            assert id > node["id"]
            cur_node = self.tree["nodes"][id]
            if cur_node["is_dummy"]:
                continue
            if node["id"] not in self._get_nondummy_ancestors(cur_node["id"]):
                return cur_node["id"]

        return sys.maxsize
            