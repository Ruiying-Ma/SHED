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
from agents.utils import CLIENT, get_result_path, get_doc_txt, pretty_repr, get_llm_response

DEBUG = False


system_prompt = """You are given a document. Your task is to extract and organize its Table of Contents (ToC).

Instructions:

1. Identify **all headings at every level** in the document (e.g., title, sections, subsections, sub-subsections, and any deeper levels).
2. **Use the exact wording of each heading as it appears** — do NOT paraphrase, shorten, or modify.
3. Preserve the original capitalization, punctuation, and numbering.
4. Infer the hierarchy levels based on formatting, numbering, or context.
5. Maintain the original order of appearance.
6. Do NOT invent, merge, or omit any headings.
7. Exclude any text that is not a heading.

Formatting requirements:

* Use the following Markdown format to represent hierarchy:

  ```markdown
  # Title  
  ## Section Header  
  ### Subsection Header  
  #### Sub-subsection Header  
  ```

* The number of `#` symbols indicates the hierarchy level.

* Include exactly one space between the `#` symbols and the header text.

* Extend to deeper levels as needed (e.g., `#####`, `######`, etc.).

* Do NOT skip levels (e.g., do not jump from `#` to `###`).

* Output only the Table of Contents in Markdown format.

* Do NOT include any explanations or extra text."""


def llm_txt_sht_gen_per_doc(dataset, filename, toc_gen_model):
    doc_txt = get_doc_txt(dataset, filename)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"DOCUMENT:\n{doc_txt}\n\nTOC IN MARKDOWN:"}
    ]

    if DEBUG:
        # print(messages)
        print(pretty_repr(messages))
        return
    
    toc_gen_response_path = str(get_result_path(dataset, toc_gen_model, "baseline", "llm_txt_sht")).replace("/core/", "/llm_gen_toc_response/").replace("baseline", "llm_txt")
    os.makedirs(os.path.dirname(toc_gen_response_path), exist_ok=True)

    llm_response = get_llm_response(
        messages = messages,
        model = toc_gen_model,
    )

    llm_response['file_name'] = filename

    with open(toc_gen_response_path, 'a') as file:
        contents = json.dumps(llm_response) + "\n"
        file.write(contents)

if __name__ == "__main__":
    for dataset in DATASET_LIST[2:]:
        if dataset == "civic_rand_v1":
            start_id = 0
            end_id = 107
        elif dataset == 'finance_rand_v1':
            start_id = 75
            end_id = 100
        elif dataset == 'contract_rand_v0_1':
            start_id = 0
            end_id = 248
        elif dataset == 'qasper_rand_v1':
            start_id = 290
            end_id = 500

        toc_gen_model = "gpt-5.4"
        for id in range(start_id, end_id):
            filename = str(id)
            llm_txt_sht_gen_per_doc(dataset, filename, toc_gen_model)


        
