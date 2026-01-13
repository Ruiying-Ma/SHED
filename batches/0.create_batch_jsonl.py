import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import json
import argparse
import config
import logging
import logging_config
import tiktoken
logging.disable(level=logging.DEBUG)

SAFETY_CHECK = True

def create_qa_jobs(
    dataset,
    context_path, # path to the generated context
    max_answer_tokens=100, 
    model="gpt-4o-mini"
):
    assert dataset in context_path
    data_root_dir = os.path.dirname(os.path.abspath(__file__)).replace("batches", "data")
    queries_path = os.path.join(data_root_dir, dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries_info = json.load(file)
    dst = context_path.replace("context.jsonl", "qa_job.jsonl")
    assert not os.path.exists(dst), dst
    assert os.path.exists(os.path.dirname(dst)), dst

    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    job_num = 0
    input_token_num = 0
    with open(context_path, 'r') as file:
        for l in file:
            context_info = json.loads(l)
            query_id = context_info["id"]
            prompt_template = queries_info[query_id]["prompt_template"]
            assert prompt_template.count("{context}") == 1
            prompt = prompt_template.replace("{context}", context_info["context"])
            job = {
                "custom_id": str(context_info["id"]),
                "method": "POST",
                # "url": "/v1/chat/completions",
                "url": "/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "max_tokens": max_answer_tokens,
                }
            }
            if SAFETY_CHECK == False:
                with open(dst, 'a') as file:
                    contents = json.dumps(job) + "\n"
                    file.write(contents)
            job_num += 1
            input_token_num += len(tokenizer.encode(prompt))
    
    flag = True

    if SAFETY_CHECK == False:
        size_bytes = Path(dst).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > 200:
            print(f"⚠️ {dst} has {round(size_mb, 2)} MB")
            flag = False
        else:
            print(f"✅ {dst} has {round(size_mb, 2)} MB")

        if job_num > 100000:
            print(f"⚠️ {dst} has {job_num} requests")
            flag = False
        else:
            print(f"✅ {dst} has {job_num} requests")

        if flag == False:
            os.remove(dst)

    print(f"✅ ${round((input_token_num * 0.075 / 1000000) + (job_num * 100 * 0.3 / 100000), 6)} costs for {dst}")

    return flag    

def create_llmjudge_jobs(
    dataset,
    answer_path, # path to the generated answer
    max_answer_tokens=100, 
    model="gpt-4o-mini",
):
    '''
    Create llm_judge jobs for Qasper
    '''
    assert dataset in ["qasper", "finance"]
    assert dataset in answer_path
    data_root_dir = os.path.dirname(os.path.abspath(__file__)).replace("batches", "data")
    queries_path = os.path.join(data_root_dir, dataset, "queries.json")
    with open(queries_path, 'r') as file:
        queries = json.load(file)
    dst = answer_path.replace("answer.jsonl", "rating_job.jsonl")
    assert not os.path.exists(dst), dst
    assert os.path.exists(os.path.dirname(dst)), dst

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_judge_prompt", f"{dataset}.txt"), 'r') as file:
        prompt_template = file.read().strip()

    assert prompt_template.count("{query}") == 1
    assert prompt_template.count("{reference}") == 1
    assert prompt_template.count("{answer}") == 1

    answers = []
    with open(answer_path, 'r') as file:
        for l in file:
            answers.append(json.loads(l))

    assert len(answers) == len(queries)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    job_num = 0
    input_token_num = 0
    for answer_info, query_info in zip(answers, queries):
        assert answer_info["id"] == query_info["id"]
        my_answer = answer_info["answer"]
        if my_answer == None:
            continue
        gold_answers = query_info["answer"]
        for ga_id, gold_answer in enumerate(gold_answers):
            prompt = prompt_template.replace("{query}", query_info['query']).replace("{reference}", gold_answer).replace("{answer}", my_answer)
            job = {
                "custom_id": str(query_info["id"]) + "-" + str(ga_id),
                "method": "POST",
                # "url": "/v1/chat/completions",
                "url": "/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "max_tokens": max_answer_tokens,
                }
            }
            if SAFETY_CHECK == False:
                with open(dst, 'a') as file:
                    contents = json.dumps(job) + "\n"
                    file.write(contents)
            job_num += 1
            input_token_num += len(tokenizer.encode(prompt))

    flag = True

    if SAFETY_CHECK == False:
        size_bytes = Path(dst).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > 200:
            print(f"⚠️ {dst} has {round(size_mb, 2)} MB")
            flag = False
        else:
            print(f"✅ {dst} has {round(size_mb, 2)} MB")

        if job_num > 100000:
            print(f"⚠️ {dst} has {job_num} requests")
            flag = False
        else:
            print(f"✅ {dst} has {job_num} requests")

        if flag == False:
            os.remove(dst)

    print(f"✅ ${round((input_token_num * 0.075 / 1000000) + (job_num * 100 * 0.3 / 100000), 6)} costs for {dst}")

    return flag    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["qasper", "civic", "contract", "finance"])
    parser.add_argument("--job", type=str, required=True, choices=["qa", "llmjudge"])
    parser.add_argument("--safety-check", action='store_true', help="Whether to check the context configures in config.py before continue")
    args = parser.parse_args()

    SAFETY_CHECK = bool(args.safety_check)

    if args.job == "llmjudge":
        assert args.dataset in ["qasper", "finance"], "Only Qasper/FinanceBench need LLM judge."

    job_num = 0
    check_context_list = input(f"Check the context configures in config.py before continue (dataset={args.dataset}, job={args.job}, safety_check={SAFETY_CHECK})... [y/n]")
    if check_context_list.lower() != 'y':
        print("Exit")
        exit(0)
    else:
        print("Continue...")
    
    for context_config in config.CONTEXT_CONFIG_LIST:
        if context_config[0] != "graphrag":
            context_jsonl_path = config.get_config_jsonl_path(args.dataset, context_config)
            logging.info(f"Creating {args.job} jobs for {context_jsonl_path}...")
            assert os.path.exists(context_jsonl_path)
        else:
            context_jsonl_path = os.path.join(config.DATA_ROOT_FOLDER, args.dataset, "baselines", "graphrag", "context.jsonl")
            assert not os.path.exists(context_jsonl_path)
        job_num += 1
        if args.job == "qa":
            create_qa_jobs(
                dataset=args.dataset,
                context_path=context_jsonl_path,
                max_answer_tokens=100,
                model="gpt-4o-mini"
            )
        else:
            assert args.job == "llmjudge"
            answer_path = context_jsonl_path.replace("context.jsonl", "answer.jsonl")
            assert os.path.exists(answer_path), answer_path
            create_llmjudge_jobs(
                dataset=args.dataset,
                answer_path=answer_path,
                max_answer_tokens=100,
                model="gpt-4o-mini"
            )
    
    print(f"Created {job_num} jobs on local machine!")