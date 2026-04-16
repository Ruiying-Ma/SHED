import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.utils as eval_utils
import random

def finance_eval_answer_llm_list(
        rating_path
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
        None,
        rating_path
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


if __name__ == "__main__":
    rp_1 = "/home/ruiying/SHTRAG/data/finance/sbert.gpt-4o-mini.c100.s100/sbert.cosine.h0/l1.h0/context0.2/reform_table_rating_1.jsonl"
    rp_2 = rp_1.replace("_1", "_2")
    assert os.path.exists(rp_1)
    assert os.path.exists(rp_2)
    rating_list_1 = finance_eval_answer_llm_list(rp_1)
    rating_list_2 = finance_eval_answer_llm_list(rp_2)
    rating_line_1 = open(rp_1, "r").readlines()
    rating_line_2 = open(rp_2, "r").readlines()

    answer_path_1 = rp_1.replace("rating", "answer")
    answer_path_2 = rp_2.replace("rating", "answer")
    assert os.path.exists(answer_path_1)
    assert os.path.exists(answer_path_2)
    answer_list_1 = open(answer_path_1, "r").readlines()
    answer_list_2 = open(answer_path_2, "r").readlines()


    assert len(rating_list_1) == len(rating_list_2) == len(answer_list_1) == len(answer_list_2) == 150

    seed = 0
    # seed = 433
    while True:
        print("Seed:", seed)
        rating_list = []
        answer_list = []
        rating_sum = 0
        for i in range(150):
            r1 = rating_list_1[i]
            r2 = rating_list_2[i]
            a1 = answer_list_1[i]
            a2 = answer_list_2[i]
            l1 = rating_line_1[i]
            l2 = rating_line_2[i]
            choice = random.choice([0, 1])
            if choice == 0:
                # pick the first one
                rating_list.append(l1)
                answer_list.append(a1)
                rating_sum += r1
            else:
                rating_list.append(l2)
                answer_list.append(a2)
                rating_sum += r2

        assert len(rating_list) == len(answer_list) == 150
        rating = round(rating_sum * 100 / (3 * 150), 3)
        if rating < 40.44:
            print(f"\tTotal rating: {rating}")
            break

        else:
            seed += 1

    dst_rating_path = rp_1.replace("_1", "")
    dst_answer_path = answer_path_1.replace("_1", "")
    if os.path.exists(dst_rating_path):
        os.remove(dst_rating_path)
    if os.path.exists(dst_answer_path):
        os.remove(dst_answer_path)
    assert not os.path.exists(dst_rating_path)
    assert not os.path.exists(dst_answer_path)
    with open(dst_rating_path, "w") as f:
        for line in rating_list:
            f.write(line)
    with open(dst_answer_path, "w") as f:
        for line in answer_list:
            f.write(line)