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
import fitz
from structured_rag import split_text_into_sentences
import numpy as np

SAFETY_CHECK = True
PROMPT_LEN_UPPER_BOUND = 126000

def _get_doc_txt(pdf_path):
    txt = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        txt += page.get_text()
    return txt

def create_tocgen_jobs(
    dataset,
    model="gpt-4o-mini"
):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    pdf_folder = os.path.join(config.DATA_ROOT_FOLDER, dataset, "pdf")
    dst = os.path.join(config.DATA_ROOT_FOLDER, dataset, "llm_txt", "llm_txt", f"sht_job.jsonl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    job_num = 0
    input_token_num = 0
    for pdf_id, pdf_name in enumerate(sorted(os.listdir(pdf_folder))):
        print(pdf_name)
        pdf_path = os.path.join(pdf_folder, pdf_name)
        with open("txt_toc_gen_prompt.txt", 'r') as file:
            TOC_GEN_PROMPT = file.read()
        assert TOC_GEN_PROMPT.count("{doc}") == 1
        doc_txt = _get_doc_txt(pdf_path)
        prompt = TOC_GEN_PROMPT.replace("{doc}", doc_txt)
        prompt_len = len(tokenizer.encode(prompt))
        
        if prompt_len < PROMPT_LEN_UPPER_BOUND:
            jobs = [{
                "custom_id": str(pdf_id),
                "method": "POST",
                # "url": "/v1/chat/completions",
                "url": "/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                    }],
                }
            }]
        else:
            chunk_num = np.ceil(prompt_len / PROMPT_LEN_UPPER_BOUND).astype(int)
            chunk_len = np.ceil(len(doc_txt) / chunk_num).astype(int)
            assert chunk_len > 0    
            jobs = []
            cur_idx = 0
            prompt_len = 0
            while cur_idx < len(doc_txt):
                print(f"\t{len(jobs)}: chunk at idx {cur_idx}")
                last_idx = min(cur_idx + chunk_len, len(doc_txt))
                chunk_txt = doc_txt[cur_idx: last_idx]
                if cur_idx == 0:
                    chunk_txt = chunk_txt + "\n... (omit the rest of the document)"
                elif cur_idx + chunk_len >= len(doc_txt):
                    chunk_txt = "... (omit the beginning of the document)\n" + chunk_txt
                else:
                    chunk_txt = "... (omit the beginning of the document)\n" + chunk_txt + "\n... (omit the rest of the document)"
                prompt = TOC_GEN_PROMPT.replace("{doc}", chunk_txt)
                job = {
                    "custom_id": str(pdf_id) + "-" + str(len(jobs)),
                    "method": "POST",
                    # "url": "/v1/chat/completions",
                    "url": "/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": prompt,
                        }],
                    }
                }
                cur_prompt_len = len(tokenizer.encode(prompt))
                assert cur_prompt_len < PROMPT_LEN_UPPER_BOUND
                prompt_len += cur_prompt_len
                jobs.append(job)
                cur_idx += chunk_len

        if SAFETY_CHECK == False:
            with open(dst, 'a') as file:
                for job in jobs:
                    contents = json.dumps(job) + "\n"
                    file.write(contents)
        job_num += len(jobs)
        input_token_num += prompt_len
    
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
    parser.add_argument("--safety-check", action='store_true', help="Whether to check the context configures in config.py before continue")
    args = parser.parse_args()

    SAFETY_CHECK = bool(args.safety_check)

    create_tocgen_jobs(
        dataset=args.dataset,
        model="gpt-4o-mini"
    )