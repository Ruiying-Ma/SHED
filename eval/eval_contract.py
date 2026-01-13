import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.utils as eval_utils

def contract_eval_answer_list(
    context_config
):
    '''
    Evaluate answer accuracy for contract-nli. 

    Filter out queries whose gold answers are NotMentioned.
    '''
    def normalize_answer(s: str):
        if s == None:
            return ""
        new_s = eval_utils.global_normalize_answer(s)
        choices = ["entailment", "contradiction"]
        ans = [c for c in choices if c in new_s]
        if len(ans) > 1:
            return ""
        assert len(ans) <= 1, new_s
        if len(ans) == 0:
            return ""
        else:
            return ans[0]

    def is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        assert my_answer in ["entailment", "contradiction", ""]
        assert gold_answer in ["entailment", "contradiction"]
        return my_answer == gold_answer

    gold_answers = eval_utils.get_gold_answers("contract")

    my_answers = eval_utils.get_answers(
        "contract",
        context_config
    )

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 1241

    accuracy_list = []
    answer_id_list = []
    for gold_answer, my_answer in zip(gold_answers, my_answers):
        assert gold_answer["id"] == my_answer["id"]
        if gold_answer["answer"] == "NotMentioned":
            continue
        answer_id_list.append(gold_answer["id"])
        if is_correct_answer(my_answer=normalize_answer(my_answer["answer"]), gold_answer=normalize_answer(gold_answer["answer"])):
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)
    assert len(accuracy_list) == len(answer_id_list) == 669
    return accuracy_list, answer_id_list

    

def contract_eval_answer(
    context_config
):
    accuracy_list, answer_id_list = contract_eval_answer_list(context_config)
    assert len(accuracy_list) == len(answer_id_list) == 669
    n_correct = sum(accuracy_list)
    n_ans = len(accuracy_list)

    print(f"ContractNLI: {n_correct} correct answers among {n_ans} queries\n\taccuracy={round(n_correct * 100 / n_ans, 3)}")

    return round(n_correct * 100 / n_ans, 3)