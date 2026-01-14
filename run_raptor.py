import json
import os
import sys
from typing import Dict 
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel, Node
from openai import OpenAI, AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging
import logging_config
import time
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from structured_rag import get_context_len
from dotenv import load_dotenv

class CustomQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        raise ValueError("should not achieve this part")

class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):
        raise ValueError("should not achieve this part")

class GPT4oMiniSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.user_prompt_template = "Please summarize the following text in no more than four sentences. Ensure that the summary includes all key details.\n\n[Start of the text]\n{text}\n[End of the text]"
        self.stats = {
            "input_tokens": 0,
            "output_tokens": 0,
            "time": 0.0
        }
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):

        try:
            logging.info(f"{self.model} summarizing (in {max_tokens} tokens) text: {context[0:min(10, len(context))]}...")
            
            openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raptor", ".env")
            assert os.path.exists(openai_key_path)
            load_dotenv(openai_key_path)
            # client = OpenAI(api_key=os.getenv("API_KEY"))
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_version=os.getenv("AZURE_API_VERSION"),
                api_key=os.getenv("AZURE_API_KEY"),
            )


            user_prompt = self.user_prompt_template.format(text=context)

            start_time = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=max_tokens,
            )
            end_time = time.time()

            summary = response.choices[0].message.content
            self.stats["input_tokens"] += len(self.tokenizer.encode(user_prompt))
            self.stats["output_tokens"] += len(self.tokenizer.encode(summary))
            self.stats["time"] += end_time - start_time

            return summary

        except Exception as e:
            print(e)
            return e

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        pass

    def create_embedding(self, text):
        raise ValueError("should not achieve this part")

class MySBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)
        self.time = 0.0

    def create_embedding(self, text):
        start = time.time()
        embedding = self.model.encode(text)
        end = time.time()
        self.time += (end - start)
        return embedding
    
class MyDPRContextEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/dpr-ctx_encoder-multiset-base"):
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
        self.time = 0.0

    def create_embedding(self, text):
        start = time.time()
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=512)["input_ids"]
        embedding = self.model(input_ids).pooler_output.detach().numpy().flatten().tolist()
        end = time.time()
        self.time += (end - start)
        return embedding


class MyDPRQueryEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/dpr-question_encoder-multiset-base"):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.time = 0.0

    def create_embedding(self, text) -> Dict:
        start = time.time()
        input_ids = self.tokenizer(text, return_tensors="pt", max_length=512)["input_ids"]
        embedding = self.model(input_ids).pooler_output.detach().numpy().flatten().tolist()
        end = time.time()
        self.time += end - start
        return embedding


class MyTE3SmallEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = None
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "raptor", '.env'))
        # self.client = OpenAI(api_key=os.getenv("API_KEY"))
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )
        self.time = 0.0

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        assert self.client is not None
        start = time.time()
        embedding = self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding
        end = time.time()
        self.time += (end - start)
        return embedding


def get_text(dataset, name: str):
    '''
    Get the full text of a document. 

    The full text is fetched from stored SHT.
    '''

    one_sht_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht", name+".json")

    with open(one_sht_path, 'r') as file:
        sht = json.load(file)
    assert "full_text" in sht
    return sht["full_text"].strip()


def build(
    dataset: str,
    name: str,
    embedding_model: str
) -> Dict:
    '''
    Store raptor tree in .pkl form.

    Returns:
        - stats (Dict):
            - input_tokens (int)
            - output_tokens (int)
            - summarization_time (float)
            - embedding_time (float)
    '''
    raptor_tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", "raptor_tree", name+".pkl")

    custom_summarizer = GPT4oMiniSummarizationModel()
    if embedding_model == "sbert":
        custom_embedder = MySBertEmbeddingModel()
    elif embedding_model == "dpr":
        custom_embedder = MyDPRContextEmbeddingModel()
    elif embedding_model == "te3small":
        custom_embedder = MyTE3SmallEmbeddingModel()
    custom_qa = CustomQAModel()

    

    if os.path.exists(raptor_tree_path):
        custom_RAConfig = RetrievalAugmentationConfig(
            summarization_model=custom_summarizer,
            qa_model=custom_qa,
            embedding_model=CustomEmbeddingModel(),
        )
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=raptor_tree_path)
        raptor_tree_nodes = RA.tree.all_nodes # Dict: index -> Node
        logging.info(f"Loaded existing raptor tree from {raptor_tree_path}.")
        logging.info(f"Existed embeddings are {list(list(raptor_tree_nodes.values())[0].embeddings.keys())}.")
        for node_index, node in raptor_tree_nodes.items():
            assert isinstance(node_index, int)
            assert isinstance(node, Node)
            assert len(node.embeddings) > 0
            if len(node.embeddings) == 1 and "EMB" in node.embeddings:
                assert embedding_model == "sbert"
                node.embeddings["sbert"] = node.embeddings.pop("EMB")
                assert 'sbert' in node.embeddings and len(node.embeddings) == 1
            if embedding_model not in node.embeddings:
                assert embedding_model != 'sbert'
                assert "sbert" in node.embeddings
                if embedding_model == "te3small":
                    assert 'dpr' in node.embeddings
                node.embeddings[embedding_model] = custom_embedder.create_embedding(node.text)
        logging.info(f"Added {embedding_model} embeddings to raptor tree.")
        RA.save(raptor_tree_path)


    else:
        os.makedirs(os.path.dirname(raptor_tree_path), exist_ok=True)
        custom_RAConfig = RetrievalAugmentationConfig(
            summarization_model=custom_summarizer,
            qa_model=custom_qa,
            embedding_model=MySBertEmbeddingModel()
        )
        RA = RetrievalAugmentation(config=custom_RAConfig)
        doc_text = get_text(dataset, name)
        RA.add_documents(doc_text)
        assert not os.path.exists(raptor_tree_path)
        os.makedirs(os.path.dirname(raptor_tree_path), exist_ok=True)
        RA.save(raptor_tree_path)


def index(
    dataset: str,
    name: str,
    query_embedding_model: str,
    query: str,
    query_id: int
):
    raptor_tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", "raptor_tree", name+".pkl")
    assert os.path.exists(raptor_tree_path), f"Raptor tree doesn't exist: {raptor_tree_path}"

    node_embedding_model = query_embedding_model
    summarization_model = "gpt-4o-mini"
    chunk_size = 100
    summary_len = 100
    distance_metric = 'cosine'
    is_raptor = True
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}/{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", "index.jsonl")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    custom_summarizer = CustomSummarizationModel()
    if query_embedding_model == "sbert":
        custom_embedder = MySBertEmbeddingModel()
    elif query_embedding_model == "dpr":
        custom_embedder = MyDPRQueryEmbeddingModel()
    elif query_embedding_model == "te3small":
        custom_embedder = MyTE3SmallEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        tr_embedding_model=custom_embedder,
        tr_context_embedding_model=query_embedding_model,
        tb_embedding_models={"empty": CustomEmbeddingModel()},
        tb_cluster_embedding_model="empty"
    )


    RA = RetrievalAugmentation(config=custom_RAConfig, tree=raptor_tree_path)

    top_k = len(RA.tree.all_nodes)

    _, layer_info = RA.retrieve(
        question=query,
        top_k=top_k,
        max_tokens=sys.maxsize,
        collapse_tree=True,
        return_layer_information=True,
    )

    assert len(layer_info) == len(RA.tree.all_nodes)

    index_info = {
        "id": int(query_id),
        "indexes": [info["node_index"] for info in layer_info],
    }

    with open(index_path, 'a') as file:
        file.write(json.dumps(index_info) + "\n")


def generate_context(
    dataset: str,
    query_embedding_model: str,
    is_ordered: bool,
    is_raptor: bool,
    context_len,
):
    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries = json.load(file)

    node_embedding_model = query_embedding_model
    summarization_model = "gpt-4o-mini"
    chunk_size = 100
    summary_len = 100
    distance_metric = 'cosine'
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}/{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", "index.jsonl")
    assert os.path.exists(index_path)
    indexes = []
    with open(index_path, 'r') as file:
        for l in file:
            indexes.append(json.loads(l))

    context_path = os.path.join(os.path.dirname(index_path), f'o{int(is_ordered)}', f"context{context_len}", "context.jsonl")
    os.makedirs(os.path.dirname(context_path), exist_ok=True)

    assert len(queries) == len(indexes)

    existing_query_ids = list()
    if os.path.exists(context_path):
        with open(context_path, 'r') as file:
            for l in file:
                context_info = json.loads(l)
                query_id = context_info["id"]
                assert query_id not in existing_query_ids
                existing_query_ids.append(context_info["id"])

    for query, index_info in zip(queries, indexes):
        
        assert query["id"] == index_info["id"]

        if query["id"] in existing_query_ids:
            print(f"\tquery_id {query_id} already has context, skip!")
            continue


        name = query["file_name"]

        true_context_len = get_context_len(
            context_ratio=context_len,
            dataset=dataset,
            sht_json_filename=name,
            min_context_len=round(max(chunk_size, summary_len) * 1.5)
        )

        raptor_tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", "raptor_tree", name+".pkl")
        assert os.path.exists(raptor_tree_path)

        custom_summarizer = CustomSummarizationModel()
        custom_embedder = CustomEmbeddingModel()
        custom_qa = CustomQAModel()

        custom_RAConfig = RetrievalAugmentationConfig(
            summarization_model=custom_summarizer,
            qa_model=custom_qa,
            embedding_model=custom_embedder,
        )

        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=raptor_tree_path)

        if not is_ordered:
            context = ""
            context_token_count = 0
            for node_index in index_info["indexes"]:
                node = RA.tree.all_nodes[int(node_index)]
                assert isinstance(node, Node)
                text = node.text
                assert text != ""
                text += "\n\n"
                text_token_count = len(tokenizer.encode(text))
                if context_token_count + text_token_count <= true_context_len:
                    context += text
                    context_token_count += text_token_count
                else:
                    break
        else:
            context_node_id = []
            context_token_count = 0
            for node_index in index_info["indexes"]:
                node = RA.tree.all_nodes[int(node_index)]
                assert isinstance(node, Node)
                text = node.text
                assert text != ""
                text += "\n\n"
                text_token_count = len(tokenizer.encode(text))
                if context_token_count + text_token_count <= true_context_len:
                    context_node_id.append(node_index)
                    context_token_count += text_token_count
                else:
                    break
            
            sorted_context_node_id = sorted(context_node_id)
            context = ""
            for node_index in sorted_context_node_id:
                node = RA.tree.all_nodes[int(node_index)]
                assert isinstance(node, Node)
                text = node.text
                assert text != ""
                text += "\n\n"
                context += text

        assert len(tokenizer.encode(context)) == context_token_count

        context_info = {
            "id": query["id"],
            "context": context,
        }
        assert query["id"] not in existing_query_ids
        existing_query_ids.append(query["id"])
        with open(context_path, 'a') as file:
            file.write(json.dumps(context_info) + "\n")
   

def raptor_num_nodes(dataset):
    import numpy as np
    custom_summarizer = CustomSummarizationModel()
    custom_embedder = MySBertEmbeddingModel()
    custom_qa = CustomQAModel()

    custom_RAConfig = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedder,
    )
    tree_dir = f"/mnt/f/Research-2024-spring/SHTRAG/database/{dataset}/raptor"
    files = os.listdir(tree_dir)
    node_count = []
    for pickle_name in files:
        assert pickle_name.endswith(".pkl")
        RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(tree_dir, pickle_name))
        node_count.append(len(RA.tree.all_nodes))
    assert len(files) == len(node_count)
    print(np.mean(node_count))


if __name__ == "__main__":
    # queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qasper", "queries.json")

    # with open(queries_path, 'r') as file:
    #     queries = json.load(file)
    
    # for query_embedding_model in ["sbert", "dpr", "te3small"]:
    #     for query_info in queries:
    #         index(
    #             dataset="qasper",
    #             name=query_info["file_name"],
    #             query_embedding_model=query_embedding_model,
    #             query=query_info["query"],
    #             query_id=query_info["id"]
    #         )
            
    # for query_embedding_model in ["sbert", "dpr", "te3small"]:
    #     for is_ordered in [True, False]:
    #         generate_context(
    #             dataset="qasper",
    #             query_embedding_model=query_embedding_model,
    #             is_ordered=is_ordered,
    #             is_raptor=False
    #         )

            
    ############################### raptor build tree
    # import config
    # from datetime import datetime
    # dataset = "finance"
    # root_dir = os.path.join(config.DATA_ROOT_FOLDER, dataset)
    # assert os.path.exists(root_dir)
    # logging.disable(logging.DEBUG)

    # for pdf_filename in sorted(os.listdir(os.path.join(root_dir, "pdf"))):
    #     pdf_path = os.path.join(root_dir, "pdf", pdf_filename)
    #     assert os.path.exists(pdf_path)
    #     cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     print(f"[{cur_time}] Processing {pdf_filename}...")
    #     try:
    #         for node_embedding_model in ["sbert", "sbert", "dpr", "te3small"]:
    #             build(
    #                 dataset=dataset,
    #                 name=pdf_filename.replace(".pdf", ""),
    #                 embedding_model=node_embedding_model
    #             )
    #             cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #             print(f"\t[{cur_time}]✅ Finished processing {pdf_filename} with {node_embedding_model}")
    #     except Exception as e:
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]❌ Error processing {pdf_filename} with {node_embedding_model}: {str(e)}")
    #         continue







    ######################### raptor index
    import config
    from datetime import datetime
    dataset = "finance"
    root_dir = os.path.join(config.DATA_ROOT_FOLDER, dataset)
    assert os.path.exists(root_dir)
    logging.disable(logging.DEBUG)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json"), 'r') as file:
        qinfo_list = json.load(file)


    for qinfo in qinfo_list[1:]:
        filename = qinfo["file_name"]
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{cur_time}] Processing {qinfo['id']}...")
        try:
            for node_embedding_model in ["sbert", "dpr", "te3small"]:
                raptor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", "raptor_tree", f"{filename}.pkl")
                assert os.path.exists(raptor_path), f"No SHT: {raptor_path} doesn't exist"
                index(
                    dataset=dataset,
                    name=filename,
                    query_embedding_model=node_embedding_model,
                    query=qinfo["query"],
                    query_id=qinfo["id"],
                )
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]✅ Finished processing {qinfo['id']} with {node_embedding_model}")
        except Exception as e:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]❌ Error processing {qinfo['id']} with {node_embedding_model}: {str(e)}")
            continue

