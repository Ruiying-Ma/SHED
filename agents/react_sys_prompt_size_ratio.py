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
from agents.utils import get_result_path, llm, get_doc_txt, get_toc_numbered
from agents.react_agent import get_agent_system_message

def get_sys_size_ratio(dataset, filename, sys_prompt):
    doc_txt = get_doc_txt(dataset, filename)
    return len(sys_prompt) / len(doc_txt)


if __name__ == "__main__":
    for dataset in DATASET_LIST:
        sys_prompt = get_agent_system_message(dataset).content
        pdf_folder = Path(DATA_ROOT_FOLDER) / dataset / "pdf"
        sys_size_ratio_list = []
        for pdf_file in pdf_folder.iterdir():
            if pdf_file.suffix.lower() == ".pdf":
                sys_size_ratio = get_sys_size_ratio(dataset, pdf_file.stem, sys_prompt)
                sys_size_ratio_list.append(sys_size_ratio)
        avg_sys_size_ratio = sum(sys_size_ratio_list) / len(sys_size_ratio_list)
        print(f"{dataset}: Average System Prompt Size Ratio = {avg_sys_size_ratio:.4f}")