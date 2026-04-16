import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from batches.utils import write_to_log
import logging
import logging_config
import argparse
import config
logging.disable(level=logging.DEBUG)


SAFETY_CHECK = True

def _upload_batch(batch_path):
    log_path_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.upload_batch_job.log", )
    batch_log_path = os.path.join(log_path_folder, os.path.basename(batch_path).replace(".jsonl", ".batch_id.log"))
    os.makedirs(os.path.dirname(batch_log_path), exist_ok=True)
    print(batch_log_path)

    MAX_REQUESTS = 100000
    MAX_FILE_SIZE_MB = 200

    # Get file size in MB
    file_size_mb = os.path.getsize(batch_path) / (1024 * 1024)

    # Check file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise Exception(f"Batch file size {file_size_mb} is greater than {MAX_FILE_SIZE_MB} MB.")

    # Check number of lines
    with open(batch_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)

    if line_count > MAX_REQUESTS:
        raise Exception(f"Batch File has more than {MAX_REQUESTS} requests: {line_count}")
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch",
    )

    
    file_id = batch_input_file.id
    write_to_log(
        log_path=batch_log_path,
        log_entry="Upload\n"+json.dumps(batch_input_file.model_dump())+"\n\n" + f"file_id: {file_id}\n\n"
    )

    print("The id of the uploaded file is", file_id)
    return file_id


def create_batch_job(batch_path, descrip):
    
    batch_input_file_id = _upload_batch(batch_path)

    log_path_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.upload_batch_job.log", )
    batch_log_path = os.path.join(log_path_folder, os.path.basename(batch_path).replace(".jsonl", ".batch_id.log"))
    os.makedirs(os.path.dirname(batch_log_path), exist_ok=True)
    print(batch_log_path)
    
    openai_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        # endpoint="/v1/chat/completions",
        endpoint="/chat/completions",
        completion_window="24h",
        metadata={
            "description": descrip,
        }
    )
    batch_id = batch_job.id
    write_to_log(
        log_path=batch_log_path,
        log_entry="Create\n"+json.dumps(batch_job.model_dump())+"\n\n" + f"batch_id: {batch_id}\n\n"
    )
    print('The id of the created is', batch_id)
    return batch_id

if __name__ == "__main__":
    for id in range(3):
        batch_jsonl_path = f"/mnt/c/users/ruiyi/downloads/paper_batch_{id + 1}.jsonl"
        assert os.path.exists(batch_jsonl_path)
        create_batch_job(
            batch_path=batch_jsonl_path,
            descrip=batch_jsonl_path,
        )