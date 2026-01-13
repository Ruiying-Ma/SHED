import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from batches.utils import write_to_log
import time
from pathlib import Path
import logging
import logging_config
import config
import argparse
logging.disable(level=logging.DEBUG)

SAFETY_CHECK = True


def _cancel_batch(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    client.batches.cancel(batch_id)

def _delete_file(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    batch_input_file_id = job.input_file_id
    output_file_id = job.output_file_id
    error_file_id = job.error_file_id
    
    
    if output_file_id:
        client.files.delete(output_file_id)
    if error_file_id:
        client.files.delete(error_file_id)
    else:
        print("No errors...")
    if batch_input_file_id:
        client.files.delete(batch_input_file_id)
    print(f"Successfully deleted {batch_id}!")


def delete_all_batch_response():
    if SAFETY_CHECK == True:
        print("Safety check!")

    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    FILE_ID_LIST = [f.id for f in list(client.files.list())] # test if the key is valid
    def is_valid_batch_job(job):
        if job.input_file_id not in FILE_ID_LIST:
            assert job.output_file_id not in FILE_ID_LIST
            assert job.error_file_id not in FILE_ID_LIST
            return False
        return True

    batch_job_list = [b for b in list(client.batches.list()) if is_valid_batch_job(b) == True]
    print(f"Retrieving the response of {len(batch_job_list)} batch jobs...")

    job_num = 0
    deleted_job_num = 0
    for job in batch_job_list:
        assert is_valid_batch_job(job) == True
        
        job_num += 1
        descrip = job.metadata.get("description", None)
        assert isinstance(descrip, str)
        job_path = descrip
        if not descrip.startswith("/home/ruiying/SHTRAG/data/"):
            job_path = os.path.join("/home/ruiying/SHTRAG/data/", descrip)
        assert os.path.exists(job_path)
        
        assert os.path.basename(job_path) in ["qa_job.jsonl", "rating_job.jsonl", "sht_job.jsonl"], job_path
        result_path = job_path.replace("_job.jsonl", "_result.jsonl")

        if not os.path.exists(result_path):
            print(f"⚠️ {result_path} does not exist, skipping...")
            continue
        
        assert os.path.exists(result_path), result_path
        deleted_job_num += 1

        input_file_id = job.input_file_id
        output_file_id = job.output_file_id
        error_file_id = job.error_file_id

        if SAFETY_CHECK == True:
            logging.info(f"Batch {job.id} (descrip={descrip}): input_file_id={input_file_id}, output_file_id={output_file_id}, error_file_id={error_file_id}")
            continue
        
        logging.info(f"Deleting batch job {job.id} (descrip={descrip})...")
        if input_file_id:
            client.files.delete(input_file_id)
            logging.info(f"Deleted input file {input_file_id}")
        else:
            logging.info("No input file...")
        if output_file_id:
            client.files.delete(output_file_id)
            logging.info(f"Deleted output file {output_file_id}")
        else:
            logging.info("No output file...")
        if error_file_id:
            client.files.delete(error_file_id)
            logging.info(f"Deleted error file {error_file_id}")
        else:
            logging.info("No errors...")


    print(f"Deleted {deleted_job_num}/{job_num} batch jobs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--safety-check", action='store_true', help="Whether to check the context configures in config.py before continue")
    args = parser.parse_args()

    SAFETY_CHECK = bool(args.safety_check)

    check_context_list = input(f"safety_check={SAFETY_CHECK}. Do you wanna continue?... [y/n]")
    if check_context_list.lower() != 'y':
        print("Exit")
        exit(0)
    else:
        print("Continue...")

    delete_all_batch_response()