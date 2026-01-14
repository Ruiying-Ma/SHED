import json
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseSummarizationModel, BaseEmbeddingModel, BaseQAModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
import os
from scipy import spatial
import numpy as np
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
import logging_config

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
    

class MyDPRQueryEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/dpr-question_encoder-multiset-base"):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.time = 0.0

    def create_embedding(self, text):
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

def index(
    dataset,
    query_embedding_model,
):      

    raptor_tree_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", "raptor_tree")

    queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries = json.load(file)

    node_embedding_model = query_embedding_model
    summarization_model = "gpt-4o-mini"
    chunk_size = 100
    summary_len = 100
    distance_metric = 'cosine'
    is_raptor = False
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "baselines", f"{node_embedding_model}.{summarization_model}.c{chunk_size}.s{summary_len}/{query_embedding_model}.{distance_metric}.raptor{int(is_raptor)}", "index.jsonl")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    existed_index_ids = list()
    if os.path.exists(index_path):
        with open(index_path, 'r') as file:
            for l in file:
                index_id = json.loads(l)["id"]
                assert f"Duplicate index id: {index_id}"
            existed_index_ids.append(index_id)

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

    for query_info in queries[1:]:
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{cur_time}] Processing {query_info['id']}...")
        
        if query_info["id"] in existed_index_ids:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]⚠️ Skipping {query_info['id']}")
            continue

        pickle_name = query_info["file_name"] + ".pkl"
        
        
        try:
            raptor_path = os.path.join(raptor_tree_dir, pickle_name)
            assert os.path.exists(raptor_path), f"No SHT: {raptor_path} doesn't exist"
        
            RA = RetrievalAugmentation(config=custom_RAConfig, tree=os.path.join(raptor_tree_dir, pickle_name))

            leaf_node_keys = RA.tree.leaf_nodes.keys() # Dict[int, Node]

            leaf_nodes = sorted([l for l in RA.tree.all_nodes.values() if l.index in leaf_node_keys], key=lambda leaf: leaf.index) # List[Node]

            assert list(range(len(leaf_nodes))) == [l.index for l in leaf_nodes]

            print(query_embedding_model, leaf_nodes[0].embeddings.keys())
            assert all([query_embedding_model in l.embeddings.keys() for l in leaf_nodes])

            embeddings = [l.embeddings[query_embedding_model] for l in leaf_nodes]

            query_embedding = custom_embedder.create_embedding(query_info["query"])

            distance_metric = spatial.distance.cosine

            distances = [distance_metric(query_embedding, embedding) for embedding in embeddings]

            indexes = np.argsort(distances)

            assert len(indexes) == len(leaf_nodes)
            assert set(indexes) == set(range(len(leaf_nodes)))

            index_info = {
                "id": query_info["id"],
                "indexes": [int(i) for i in indexes]
            }
            with open(index_path, 'a') as file:
                file.write(json.dumps(index_info) + "\n")

            existed_index_ids.append(query_info['id'])

            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]✅ Finished processing {query_info['id']} with {node_embedding_model}")

        except Exception as e:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]❌ Error processing {query_info['id']} with {node_embedding_model}: {str(e)}")
            continue



if __name__ == "__main__":
    for query_embedding_model in ["sbert", "dpr", "te3small"]:
        index(
            dataset='finance',
            query_embedding_model=query_embedding_model
        )