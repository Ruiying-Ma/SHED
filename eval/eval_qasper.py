import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.utils as eval_utils
from collections import Counter
import re

def qasper_eval_answer_f1(
        context_config
):
    def normalize_answer(s: str):
        return eval_utils.white_space_fix(eval_utils.remove_articles(eval_utils.remove_punc(eval_utils.lower(s))))

    def token_f1(my_answer: str, gold_answer: str) -> float:
        if my_answer == None:
            return 0.0
        my_answer_tokens = normalize_answer(my_answer).split()
        gold_answer_tokens = normalize_answer(gold_answer).split()
        common = Counter(my_answer_tokens) & Counter(gold_answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(my_answer_tokens)
        recall = 1.0 * num_same / len(gold_answer_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    gold_answers = eval_utils.get_gold_answers("qasper")
    my_answers = eval_utils.get_answers(
        "qasper",
        context_config
    )

    assert len(my_answers) == len(gold_answers)
    assert len(gold_answers) == 1451
    
    f1 = 0.0
    n_answer = 0
    for gold_answer, my_answer in zip(gold_answers, my_answers):
        assert gold_answer["id"] == my_answer["id"]
        if len(gold_answer['answer']) == 0:
            continue
        f1 += max([token_f1(my_answer=my_answer["answer"], gold_answer=ga) for ga in gold_answer["answer"]])
        n_answer += 1

    print(f"Qasper: among {n_answer} queries\n\tf1={round(f1  * 100 / n_answer, 3)}")

    return round(f1 * 100 / n_answer, 3)


def qasper_eval_answer_llm_list(
        context_config
):
    def extract_rating_from_gpt_response(s: str):
        if not isinstance(s, str):
            return 0
        match = re.search(r"Rating:\s*\[\[(\d)\]\]", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*\[(\d)\]", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*(\d)", s)
        if match:
            return int(match.group(1))
        match = re.search(r"Rating:\s*\*\*(\d)\*\*", s)
        if match:
            return int(match.group(1))
        return 0

    ratings = eval_utils.get_ratings(
        "qasper",
        context_config
    )

    assert len(ratings) == 1451
    
    rating_list = []
    for rating_info in ratings:
        candid_ratings = [extract_rating_from_gpt_response(r) for r in rating_info["rating"]]
        if len(candid_ratings) == 0:
            rating = 0
        else:
            rating = max(candid_ratings)
        rating_list.append(rating)
    assert len(rating_list) == 1451
    return rating_list


def qasper_eval_answer_llm(context_config):
    rating_list = qasper_eval_answer_llm_list(context_config)
    assert len(rating_list) == 1451
    tot_rating = sum(rating_list)
    n_answer = len(rating_list)
    print(f"Qasper: among {n_answer} queries\n\trating={round(tot_rating * 100 / (3 * n_answer), 3)}")

    return round(tot_rating * 100 / (3 * n_answer), 3)

if __name__ == "__main__":
    embedding_model = "sbert"
    qasper_eval_answer_llm(
        chunk_size=100,
        summary_len=100,
        node_embedding_model=embedding_model,
        query_embedding_model=embedding_model,
        summarization_model="gpt-4o-mini",
        embed_hierarchy=True,
        distance_metric="cosine",
        context_hierarchy=True,
        context_raw=True,
        context_len=1000,
        is_intrinsic=False,
        is_baseline=False,
        is_raptor=False,
        is_ordered=False,
        is_grobid=False
    )
    