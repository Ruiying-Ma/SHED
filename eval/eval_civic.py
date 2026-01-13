import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.utils as eval_utils
from typing import List
import json
import re
import config

def civic_q1_eval_answer_list(
    context_config
):
    '''
    Evaluate the accuracy of Civic q1.
    '''
    def normalize_answer(s: str):
        return eval_utils.global_normalize_answer(s)

    def is_correct_answer(my_answer: str, gold_answer: str) -> bool:
        if not isinstance(my_answer, str):
            return False
        assert isinstance(my_answer, str)
        assert isinstance(gold_answer, str)
        delimiters = ["-", "/"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        split_gold_answer = re.split(regex_pattern, gold_answer)
        return all([normalize_answer(ga.strip()) in normalize_answer(my_answer) for ga in split_gold_answer])
    
    
    my_answers = eval_utils.get_answers(
        "civic",
        context_config
    )

    gold_answers = eval_utils.get_gold_answers("civic")

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 380 + 38

    accuracy_list = [
        1 if is_correct_answer(my_answer=my_answer["answer"], gold_answer=gold_answer["answer"]) == True else 0
        for my_answer, gold_answer in zip(my_answers[:380], gold_answers[:380])
    ]
    assert len(accuracy_list) == 380
    return accuracy_list


def civic_q1_eval_answer(context_config):
    answer_accuracy = civic_q1_eval_answer_list(context_config)
    assert len(answer_accuracy) == 380
    n_correct = sum(answer_accuracy)
    print(f"Civic q1: {n_correct} correct answers among 380 queries\n\taccuracy={round(n_correct * 100 / 380, 3)}")
    return round(n_correct * 100 / 380, 3)


def civic_q2_eval_answer_list(
    context_config
):
    '''
    Evaluate the recall, precision and f1 of Civic q2.

    Corner cases:
    https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    '''

    def normalize_project(s: str):
        pattern = r"\(.*?\)"
        new_s = re.sub(pattern, '', s)
        return eval_utils.white_space_fix(eval_utils.remove_punc(eval_utils.lower(eval_utils.normalize_string(new_s))))
    
    def my_projects(my_answer: str, projects: list):
        if my_answer == None:
            return set()
        raw_my_projects = my_answer.replace("[", "").replace("]", "").split(",")
        candid_my_projects = [normalize_project(p.strip()) for p in raw_my_projects]
        my_projects = set(candid_my_projects).intersection(set(projects))
        return my_projects
    
    def gold_projects(gold_answer: list):
        return set([normalize_project(p.strip()) for p in gold_answer])
    
    def eval(my_answer: set, gold_answer: set):
        assert isinstance(my_answer, set)
        assert isinstance(gold_answer, set)
        assert all(isinstance(ma, str) for ma in my_answer)
        assert all(isinstance(ga, str) for ga in gold_answer)
        intersection = gold_answer.intersection(my_answer)
        if len(intersection) == 0:
            if len(gold_answer) == 0 and len(my_answer) == 0:
                recall = 1.0
                precision = 1.0
                f1 = 1.0
            else:
                recall = 0.0
                precision = 0.0
                f1 = 0.0
        else:
            recall = len(intersection) / len(gold_answer)
            precision = len(intersection) / len(my_answer)
            f1 = 2 * recall * precision / (recall + precision)
        return recall, precision, f1

    
    my_answers = eval_utils.get_answers(
        "civic",
        context_config
    )

    gold_answers = eval_utils.get_gold_answers("civic")

    assert len(gold_answers) == len(my_answers)
    assert len(gold_answers) == 380 + 38

    projects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("eval", "data"), "civic", "projects.json")
    with open(projects_path, 'r') as file:
        m_file_projects = json.load(file)

    recall_list = []
    precision_list = []
    f1_list = []
    for my_answer, gold_answer in zip(my_answers[380:], gold_answers[380:]):
        assert my_answer["id"] == int(gold_answer["id"])

        file_name = gold_answer["file_name"]
        projects_list = list([normalize_project(p) for p in m_file_projects[file_name].keys()])

        recall, precision, f1 = eval(my_answer=my_projects(my_answer["answer"], projects_list), gold_answer=gold_projects(gold_answer["answer"]))
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    assert len(recall_list) == len(precision_list) == len(f1_list) == 38

    return recall_list, precision_list, f1_list
    

def civic_q2_eval_answer(context_config):
    recall_list, precision_list, f1_list = civic_q2_eval_answer_list(context_config)
    assert len(recall_list) == len(precision_list) == len(f1_list) == 38
    tot_recall = sum(recall_list)
    tot_precision = sum(precision_list)
    tot_f1 = sum(f1_list)
    print(f"Civic q2: among 38 queries\n\tavg_recall={round(tot_recall * 100/ 38, 3)}\n\tavg_precision={round(tot_precision * 100/ 38, 3)}\n\tavg_f1={round(tot_f1 * 100/ 38, 3)}")

    return round(tot_recall * 100/ 38, 3), round(tot_precision * 100 / 38, 3), round(tot_f1 * 100/ 38, 3)

if __name__ == "__main__":

    check_context_list = input(f"Check the context configures in config.py before continue... [y/n]")
    if check_context_list.lower() != 'y':
        print("Exit")
        exit(0)
    else:
        print("Continue...")

    m_context_config_result = dict()

    for context_config in config.CONTEXT_CONFIG_LIST:
        context_jsonl_path = config.get_config_jsonl_path("civic", context_config)
        assert os.path.exists(context_jsonl_path)
        print(f"Evaluating answers for {context_jsonl_path}...")
        civic_q1_eval_answer(
            context_config,
        )
        civic_q2_eval_answer(
            context_config,
        )