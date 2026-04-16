import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz  # PyMuPDF
import tiktoken
from script_python.financebench.doc_in_query import get_doc_in_query
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, BaseEmbeddingModel, BaseSummarizationModel, BaseQAModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.disable(level=logging.WARNING)


class CustomQAModel(BaseQAModel):
    def __init__(self):
        pass

    def answer_question(self, context, question):
        raise ValueError("should not achieve this part")

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        pass

    def create_embedding(self, text):
        raise ValueError("should not achieve this part")
    
class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=100, stop_sequence=None):
        raise ValueError("should not achieve this part")

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def count_tokens_from_txt(text, model):
    tokenizer = tiktoken.get_encoding(model)
    tokens = tokenizer.encode(text)
    return len(tokens)


def count_tokens_from_raptor_leaves(raptor_tree_pickle_path):
    rag_config = RetrievalAugmentationConfig(
        summarization_model=CustomSummarizationModel(),
        qa_model=CustomQAModel(),
        tr_embedding_model=CustomEmbeddingModel(),
        tr_context_embedding_model="sbert",
        tb_embedding_models={"empty": CustomEmbeddingModel()},
        tb_cluster_embedding_model="empty"
    )
    RA = RetrievalAugmentation(config=rag_config, tree=raptor_tree_pickle_path)

    leaf_node_keys = RA.tree.leaf_nodes.keys() # Dict[int, Node]
    leaf_nodes = sorted([l for l in RA.tree.all_nodes.values() if l.index in leaf_node_keys], key=lambda leaf: leaf.index) # List[Node]

    text = ""
    for leaf in leaf_nodes:
        text += leaf.text + " "

    return count_tokens_from_txt(text, model="cl100k_base")


if __name__ == "__main__":
    # raptor_tree_folder = "/home/ruiying/SHTRAG/data/civic/baselines/raptor_tree"
    # token_list = []
    # for tree_file in os.listdir(raptor_tree_folder):

    #     raptor_tree_pickle_path = os.path.join(raptor_tree_folder, tree_file)

    #     n_tokens = count_tokens_from_raptor_leaves(raptor_tree_pickle_path)

    #     print(f"{tree_file}\t{n_tokens} tokens")
    #     token_list.append(n_tokens)

    # print(f"avg_tokens = {sum(token_list)/len(token_list)}")


##################################################################
    pfd_folder = "/home/ruiying/SHTRAG/data/civic/pdf"
    sum_tokens = 0
    pdf_num = 0
    for pdf_file in sorted(os.listdir(pfd_folder)):
        pdf_path = os.path.join(pfd_folder, pdf_file)
        text = extract_text(pdf_path)
        n_tokens = count_tokens_from_txt(text, model="cl100k_base")
        print(f"{pdf_file}\t{n_tokens} tokens")
        sum_tokens += n_tokens
        pdf_num += 1

    print(f"avg_tokens = {sum_tokens/pdf_num}")
    print(f"total_tokens = {sum_tokens}")


#######################################################################

    # n_token = count_tokens_from_raptor_leaves(
    #     "/home/ruiying/SHTRAG/data/civic/baselines/raptor_tree/01262022-1835.pkl"
    # )
    # print(n_token)