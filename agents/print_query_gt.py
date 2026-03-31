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


if __name__ == "__main__":
    dataset = "qasper"
    query_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
    with open(query_path, "r") as f:
        queries = json.load(f)

    for query in queries:
        answers = '\n'.join(query['answer'])
        print(f"[Question]\n{query['query']}\n[Reference Answer]\n{answers}\n{'='*50}")

    