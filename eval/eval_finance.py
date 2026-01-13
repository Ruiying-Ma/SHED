import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.utils as eval_utils

def finance_eval_answer_llm_list(
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
        "finance",
        context_config
    )

    assert len(ratings) == 150
    
    rating_list = []
    for rating_info in ratings:
        candid_ratings = [extract_rating_from_gpt_response(r) for r in rating_info["rating"]]
        if len(candid_ratings) == 0:
            rating = 0
        else:
            rating = max(candid_ratings)
        rating_list.append(rating)

    assert len(rating_list) == 150
    return rating_list

    


def finance_eval_answer_llm(
        context_config
):
    rating_list = finance_eval_answer_llm_list(context_config)
    tot_rating = sum(rating_list)
    n_answer = len(rating_list)
    assert n_answer == 150
    print(f"finance: among {n_answer} queries\n\trating={round(tot_rating * 100 / (3 * n_answer), 3)}")

    return round(tot_rating * 100 / (3 * n_answer), 3)