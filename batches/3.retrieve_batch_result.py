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


def _retrieve_response(batch_id, result_path):
    assert not os.path.exists(result_path)
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch {batch_id} is in status {status}...")

    batch_output_file_id = job.output_file_id

    file_response = client.files.content(batch_output_file_id)

    with open(result_path, 'wb') as file:
        file.write(file_response.content)

    print(f'Batch {batch_id} has beed retrieved!')

def _retrieve_error(batch_id):
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    job = client.batches.retrieve(batch_id)
    status = job.status
    if status != "completed":
        raise ValueError(f"Batch {batch_id} is in status {status}...")

    error_file_id = job.error_file_id
    # Same check for errors
    if error_file_id:
        errors = client.files.content(error_file_id).content.decode("utf-8")
    else:
        print(f"Batch {batch_id} has no error...")
        return
    
    log_path =os.path.join (os.path.dirname((os.path.abspath(__file__))), "error.log")
    write_to_log(
        log_path=log_path,
        log_entry="Retrieve Error\n"+ str(errors) + "\n\n" + f"batch_id: {batch_id}\n\n"
    )


def retrieve_all_batch_response():
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
    failed_job_num = 0
    for job in batch_job_list:
        assert is_valid_batch_job(job) == True

        batch_id = job.id
        status = job.status
        job_num += 1
        if status != "completed":
            failed_job_num += 1
            print(f"⚠️ Batch job {batch_id} is in status {status}, skipping...")
            continue
        
        descrip = job.metadata.get("description", None)
        assert isinstance(descrip, str)
        job_path = descrip
        if not descrip.startswith("/home/ruiying/SHTRAG/data/"):
            job_path = os.path.join("/home/ruiying/SHTRAG/data/", descrip)
        assert os.path.exists(job_path)

        logging.info(f"Retrieving batch job {batch_id} (descrip={descrip})...")
        
        assert os.path.basename(job_path) in ["qa_job.jsonl", "rating_job.jsonl", "sht_job.jsonl"], job_path
        result_path = job_path.replace("_job.jsonl", "_result.jsonl")

        assert not os.path.exists(result_path), result_path
        assert os.path.exists(os.path.dirname(result_path)), result_path

        if SAFETY_CHECK == True:
            continue

        response_contents = client.files.content(job.output_file_id).content

        with open(result_path, 'wb') as file:
            file.write(response_contents)

    print(f"Retrieved {job_num} from OpenAI, {failed_job_num} failed...")


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

    retrieve_all_batch_response()