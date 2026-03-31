import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import CLIENT, get_result_path, llm


DEBUG = False

def llm_judge_per_query(dataset, qid, query, gt_answer, llm_answer, answer_gen_model, judge_model, method):
    prompt = (Path(__file__).parent / "llm_judge_prompt_finance.txt").read_text()
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": F"QUESTION:\n{query}\n\nREFERENCE_ANSWER:\n{gt_answer}\n\nCANDIDATE_ANSWER:\n{llm_answer}\n\nOUTPUT JSON:"}
    ]
    if DEBUG:
        print(messages)
        return

    response = CLIENT.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=0.0
    ).choices[0].message.content.strip()

    # try to extract score
    score_pattern = r'"score"\s*:\s*([0-9]*\.?[0-9]+)'
    score_match = re.search(score_pattern, response)
    if score_match:
        score = float(score_match.group(1))
    else:
        score = None

    # log the response for debugging
    llm_judge_response_path = str(get_result_path(dataset, answer_gen_model, method)).replace("/core/", "/llm_judge_response/")
    os.makedirs(os.path.dirname(llm_judge_response_path), exist_ok=True)
    with open(llm_judge_response_path, 'a') as file:
        contents = json.dumps({
            "id": qid,
            "score": score,
            "judge_model": judge_model,
            "query": query,
            "gt_answer": gt_answer,
            "llm_answer": llm_answer,
            "response": response,
        }) + "\n"
        file.write(contents)


    return score


if __name__ == "__main__":
    for answer_gen_model in ["gpt-5.4", "gpt-5-mini"]:
        for method in ['baseline', 'toc_in_context', 'react_agent' , 'react_agent_grep_all', 'react_agent_grep_id', 'react_agent_grep_next', 'react_agent_grep_next_notoc']:
            judge_model = "gpt-4o-mini"
            dataset = 'finance'

            print(f"LLM Judge: {dataset}")
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)

            result_jsonl_path = get_result_path(dataset, answer_gen_model, method)
            with open(result_jsonl_path, 'r') as file:
                results = [json.loads(line) for line in file]
            
            assert len(results) <= len(queries)


            score_list = []

            for result, qinfo in zip(results, queries):
                assert result['id'] == qinfo['id']
                if result['id'] < 30:
                    continue
                if result['is_success'] == False:
                    logging.warning(f"Skipping query id {qinfo['id']} due to unsuccessful answer generation")
                    continue
                assert result['is_success'] == True

                print(f"\tquery id: {qinfo['id']}")

                score = llm_judge_per_query(dataset, qinfo['id'], qinfo['query'], qinfo['answer'][0], result['message'], answer_gen_model, judge_model, method)

                if score is not None:
                    score_list.append(score)
                    print(f"query id {qinfo['id']}: {score}")
                else:
                    logging.warning(f"Score not found for query id {qinfo['id']}")

            print(f"Average score: {sum(score_list) / len(score_list) if score_list else 'N/A'}")