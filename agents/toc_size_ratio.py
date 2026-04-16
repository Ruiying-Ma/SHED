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

def get_toc_size_ratio(dataset, filename):
    doc_txt = get_doc_txt(dataset, filename)
    toc_txt = get_toc_numbered(dataset, filename)

    return len(toc_txt) / len(doc_txt)


if __name__ == "__main__":
    for dataset in DATASET_LIST:
        pdf_folder = Path(DATA_ROOT_FOLDER) / dataset / "pdf"
        toc_size_ratio_list = []
        for pdf_file in pdf_folder.iterdir():
            if pdf_file.suffix.lower() == ".pdf":
                toc_size_ratio = get_toc_size_ratio(dataset, pdf_file.stem)
                toc_size_ratio_list.append(toc_size_ratio)
        
        avg_toc_size_ratio = sum(toc_size_ratio_list) / len(toc_size_ratio_list)
        print(f"{dataset}: Average TOC Size Ratio = {avg_toc_size_ratio:.4f}")