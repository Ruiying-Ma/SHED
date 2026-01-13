import re
from typing import Dict, List
import os
import json
import tiktoken

def max_min(list: List[int], thresh: int):
    '''
    Return the maximal item in list that is smaller than thresh. If list is empty, return 0.

    Args:
        - list (List[int])
        - thresh (int)

    Return:
        - the maximal item in list that is smaller than thresh
    '''
    if (not isinstance(list, List)) or (not all([isinstance(l, int) for l in list])):
        raise ValueError("list must be a list of integers")
    if not isinstance(thresh, int):
        raise ValueError("thresh must be an integer")
    if thresh < 1:
        raise ValueError("thresh must be >= 1")
    if any([item < 1 for item in list]):
        raise ValueError("list must be a list of integers that are all >= 1")

    min_list = [item for item in list if item < thresh]
    if len(min_list) == 0:
        return 0

    return max(min_list)

def clean_space(text: str) -> str:
    '''
    Replace all spaces '\s' in `text` string with " ".
    '''
    return " ".join(re.sub(r'\s', " ", text).split())

def split_text_into_sentences(delimiters, text) -> List:
    '''
    Split `text` into sentences using `delimiters`.

    Args:
        - delimiters (list)
        - text (str)

    Return:
        - A list of sentences
    '''
    regex_pattern = f"([{re.escape(''.join(delimiters))}])"
    split_parts = re.split(regex_pattern, text)
    sentences = [split_parts[i] + split_parts[i + 1] for i in range(0, len(split_parts) - 1, 2) if len((split_parts[i] + split_parts[i+1]).strip()) > 0]
    if len(split_parts) % 2 != 0 and len(split_parts[-1].strip()) > 0:
        sentences.append(split_parts[-1])
    return sentences

def split_text_into_chunks(chunk_size, text, tokenizer) -> List:
    '''
    Split `text` into sentences, then concat sentences into chunks not exceending `chunk_size`. The concatenator is "".

    Args:
        - chunk_size (int): counted by token num
        - text (str)
        - tokenizer

    Return:
        - A list of chunks
    '''
    sentence_delimiters = [".", "!", "?", "\n"]
    sentences = split_text_into_sentences(sentence_delimiters, text)
    n_tokens = [len(tokenizer.encode(sentence)) for sentence in sentences]

    chunks = [] # list of chunks
    current_chunk = [] # current chunk = list of sentences
    current_length = 0 # current chunk size

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if len(sentence.strip()) == 0:
            continue

        if current_length + token_count <= chunk_size:
            current_chunk.append(sentence)
            current_length += token_count
        else: 
            # current_length + token_count > chunk_size
            if len(current_chunk) > 0:
                chunks.append("".join(current_chunk))
            if token_count <= chunk_size:
                current_chunk = [sentence]
                current_length = token_count
            else:
                # token_count > max_tokens
                chunks.append(sentence)
                current_chunk = []
                current_length = 0

    # Add the last chunk if it's not empty
    if len(current_chunk) > 0:
        chunks.append("".join(current_chunk))
    
    return chunks

def get_nondummy_ancestors(nodes: List[Dict], node_id: int) -> List[int]:
    '''
    Return the list of nondummy ancestors' ids of the given nondummy node in ascending order, from root to node. Note that -1 (fake root) is not contained.
    '''

    if not isinstance(node_id, int) or node_id < 0 or node_id >= len(nodes):
        raise ValueError(f"node_{node_id} doesn't exist")
    
    node = nodes[node_id]
    
    if int(node["id"]) != node_id:
        raise ValueError(f'''node_{node_id} and its id={int(node["id"])} deosn't match''')
    if node["is_dummy"]:
        raise ValueError(f"node_{node_id} is dummy")

    if "nondummy_parent" not in node:
        raise ValueError(f"node {node['id']} doesn't have nondummy_parent: {node['is_dummy']}, {node['type']}")

    cur_parent_id = node["nondummy_parent"]
    ancestors = []
    while cur_parent_id != -1:
        ancestors.append(cur_parent_id)
        cur_node = nodes[cur_parent_id]
        assert not cur_node["is_dummy"]
        cur_parent_id = cur_node["nondummy_parent"]

    assert ancestors[::-1] == sorted(ancestors)
    if len(ancestors) > 0:
        assert ancestors[0] == node["nondummy_parent"]
    return ancestors[::-1]


def get_context_len(context_ratio, dataset, sht_json_filename: str, min_context_len):
    assert dataset in ["civic", "contract", "qasper", "finance"], dataset
    if context_ratio > 1:
        return max(min_context_len, context_ratio)
    
    sht_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset,
        "sbert.gpt-4o-mini.c100.s100", 
        "sht", 
        sht_json_filename + ".json"
    )


    with open(sht_path, 'r') as file:
        sht = json.load(file)

    assert "full_text" in sht
    full_text: str = sht["full_text"]
    context_n_chars = min(int(round(len(full_text) * context_ratio)), len(full_text))
    context_len = len(tiktoken.get_encoding("cl100k_base").encode(full_text[:context_n_chars]))
    if context_len < min_context_len:
        context_len = min_context_len
    return context_len