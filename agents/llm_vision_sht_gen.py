import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
# set logging level as info
logging.basicConfig(level=logging.INFO)
import logging_config
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name
from math import ceil, floor, sqrt
from io import BytesIO
from PIL import Image

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import CLIENT, get_result_path, get_doc_txt, pretty_repr, get_llm_response, get_cost_usd
from pdf2image import convert_from_path
import base64
import fitz
import hashlib
import unicodedata
import string
import calendar

DEBUG = False

def hash8_digits(s: str) -> str:
    h = hashlib.sha256(s.encode()).hexdigest()
    num = int(h, 16)          # convert hex → integer
    return str(num % 10**8).zfill(8).lower()

def normalize_string(text: str):
    return unicodedata.normalize('NFKC', text)

def white_space_fix(text):
    return " ".join(text.split())

def replace_punc(text, rep = " "):
    exclude = set(string.punctuation)
    # replace punctuation with space
    return "".join(ch if ch not in exclude else rep for ch in text)

def lower(text):
    return text.lower()

def format_doc_name(doc_name, dataset):
    if dataset == "civic":
        # doc_name is in the format of "MMDDYYYY"
        month = int(doc_name[:2])
        day = int(doc_name[2:4])
        year = int(doc_name[4:8])
        month_name = calendar.month_name[month]
        return f"Meeting: {month_name} {day}, {year}"
    elif dataset == 'contract':
        return replace_punc(white_space_fix(normalize_string(doc_name))).capitalize()
    elif dataset == 'finance':
        # get a fixed hash of the doc_name with 8 digits
        name_hash = hash8_digits(doc_name)
        return f"Financial document {name_hash.lower()}"
    elif dataset == 'qasper':
        with open(f"/home/ruiying/SHTRAG/data/qasper/grobid/grobid/{doc_name}.json", 'r') as file:
            grobid_info = json.load(file)
        title = grobid_info['title']
        return title
    else:
        raise NotImplementedError(f"Doc name formatting for dataset {dataset} not implemented yet")


system_prompt = """You are given images of a document's pages. Your task is to extract and organize a Table of Contents (ToC) of the document from the images.

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

def get_tokens_per_image(width_pxl, height_pxl, model, detail_level):
    original_patch_cnt = ceil(width_pxl / 32) * ceil(height_pxl / 32)

    if model == 'gpt-5.4' and detail_level == 'original':
        patch_budget = 10000
    else:
        raise NotImplementedError(f"Model {model} with detail level {detail_level} not supported for token estimation.")
    

    if original_patch_cnt <= patch_budget:
        resized_patch_cnt = original_patch_cnt
    else:
        # shrink_factor = sqrt((32^2 * patch_budget) / (width * height))
        # adjusted_shrink_factor = shrink_factor * min(
        # floor(width * shrink_factor / 32) / (width * shrink_factor / 32),
        # floor(height * shrink_factor / 32) / (height * shrink_factor / 32)
        # )
        

        shrink_factor = sqrt((32**2 * patch_budget) / (width_pxl * height_pxl))
        adjusted_shrink_factor = shrink_factor * min(
            floor(width_pxl * shrink_factor / 32) / (width_pxl * shrink_factor / 32),
            floor(height_pxl * shrink_factor / 32) / (height_pxl * shrink_factor / 32)
        )

        resized_width = floor(width_pxl * adjusted_shrink_factor / 32) * 32
        resized_height = floor(height_pxl * adjusted_shrink_factor / 32) * 32

        resized_patch_cnt = ceil(resized_width / 32) * ceil(resized_height / 32)
    
    return resized_patch_cnt


def pil_to_base64_data_url(pil_image, format="PNG"):
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    base64_encoded = base64.b64encode(buffer.read()).decode("utf-8")
    mime_type = f"image/{format.lower()}"

    return f"data:{mime_type};base64,{base64_encoded}"

def create_chapter_page(
    title,
    format="PNG",
    width=595,   # default A4 width in points
    height=842,  # default A4 height in points
):
    # Create a new PDF
    doc = fitz.open()

    # Add a page with custom dimensions
    page = doc.new_page(width=width, height=height)

    # Define font size
    font_size = 36

    # Choose a standard font
    font_name = "helv"  # Helvetica

    # Compute text width & height
    text_length = fitz.get_text_length(title, fontname=font_name, fontsize=font_size)

    # Center position
    x = (width - text_length) / 2
    y = height / 2

    # Insert centered title
    page.insert_text(
        (x, y),
        title,
        fontsize=font_size,
        fontname=font_name,
    )

    # save this fitz pdf page into base64_encoded as in pil_to_base64_data_url
    pix = page.get_pixmap()

    img_bytes = pix.tobytes("png")  # actual PNG bytes
    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")


    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_encoded}"
        


# def get_images_per_pdf(pdf_path):
#     images = convert_from_path(pdf_path)
#     doc_title_img = create_chapter_page(title=format_doc_name(Path(pdf_path).stem, dataset=Path(pdf_path).parent.parent.stem))
#     data_urls = [doc_title_img] + [pil_to_base64_data_url(img) for img in images]
#     return data_urls

def get_images(image_metadata_list):
    data_urls = []
    for pdf_path, pid in image_metadata_list:
        if pid == 0:
            # this is the additional chapter page
            title = format_doc_name(Path(pdf_path).stem, dataset=Path(pdf_path).parent.parent.stem)
            data_url = create_chapter_page(title=title)
        else:
            image = convert_from_path(pdf_path, first_page=pid, last_page=pid)
            data_url = pil_to_base64_data_url(image[0])
        
        data_urls.append(data_url)
    
    return data_urls


def get_cost_per_dataset(dataset):
    pdf_folder = Path(DATA_ROOT_FOLDER) / dataset / "pdf"
    cost_info = dict()
    for pdf_file in sorted(pdf_folder.iterdir()):
        pdf_path = str(pdf_file.resolve())
        images = convert_from_path(pdf_path)
        tokens = sum([get_tokens_per_image(img.width, img.height, model='gpt-5.4', detail_level='original') for img in images])
        cost = get_cost_usd(
            model='gpt-5.4',
            input=tokens,
            cached=0,
            output=0,
        )
        cost_info[pdf_file.stem] = {
            "num_pages": len(images),
            "height": images[0].height,
            "width": images[0].width,
            "tokens": tokens,
            "cost_usd": cost
        }
        logging.info(f"{dataset}, {pdf_file.stem}, ${cost:.2f}")

    return cost_info

def get_cost():
    total_cost = dict()
    dst_path = "estimated_llm_vision_cost.json"
    for dataset in DATASET_LIST:
        cost_info = get_cost_per_dataset(dataset)
        total_cost[dataset] = cost_info
    with open(dst_path, 'w') as f:
        json.dump(total_cost, f, indent=2)


def sht_gen_per_query(dataset, qinfo, toc_gen_model):
    file_names = qinfo['new_file_names']

    image_metadata_list = []
    for file_name in file_names:
        pdf_path = str(Path(DATA_ROOT_FOLDER) / dataset.split("_")[0] / "pdf" / f"{file_name}.pdf")
        num_pages = len(fitz.open(pdf_path))
        image_metadata_list.extend([
            [pdf_path, pid]
            for pid in range(0, num_pages + 1) # 0: additional chapter page, 1~num_pages: original pages
        ])

    # image_list = []
    # for file_name in file_names:
    #     pdf_path = str(Path(DATA_ROOT_FOLDER) / dataset.split("_")[0] / "pdf" / f"{file_name}.pdf")
    #     # per_pdf_images = convert_from_path(pdf_path)
    #     # data_urls = [pil_to_base64_data_url(img) for img in images]
    #     image_list += get_images_per_pdf(pdf_path)

    if DEBUG == True:
        # save the images to local for debugging
        debug_folder = Path(__file__).parent / "debug_images" / dataset
        os.makedirs(debug_folder, exist_ok=True)
        image_list = get_images(image_metadata_list)
        print(f"Dataset: {dataset}, Query ID: {qinfo['id']}, File Name: {qinfo['file_name']}, Number of pages: {len(image_list)}")
        for idx, data_url in enumerate(image_list):
            header, encoded = data_url.split(",", 1)
            img_data = base64.b64decode(encoded)
            with open(debug_folder / f"{qinfo['file_name']}_page_{idx}.png", 'wb') as f:
                f.write(img_data)
        return

    # 50 images per batch for LLM input
    for i in range(0, len(image_metadata_list), 50):
        if dataset == 'finance_rand_v1' and qinfo['file_name'] == "29" and ((i // 50) <= 4):
            continue

        batch_data_urls = get_images(image_metadata_list[i:i+50])
    
        batch_images = [
            {"type": "image_url", "image_url": {
                "url": url}} for url in batch_data_urls
        ]
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [
                {"type": "text", "text": "Here are the images of the document's pages. TOC IN MARKDOWN:"}
            ] + batch_images}
        ]

        toc_gen_response_path = str(get_result_path(dataset, toc_gen_model, "baseline", "llm_vision_sht")).replace("/core/", "/llm_gen_toc_response/").replace("baseline", "llm_vision")
        os.makedirs(os.path.dirname(toc_gen_response_path), exist_ok=True)

        llm_response = get_llm_response(
            messages = messages,
            model = toc_gen_model,
        )
    
        llm_response['file_name'] = qinfo['file_name']
        llm_response['batch_id'] = i // 50

        with open(toc_gen_response_path, 'a') as file:
            contents = json.dumps(llm_response) + "\n"
            file.write(contents)


if __name__ == "__main__":
    for dataset in DATASET_LIST[2:]:
    # for dataset in ['fiannce_rand_v1']:
        if dataset == "civic_rand_v1":
            start_id = 0
            end_id = 107
        elif dataset == 'finance_rand_v1':
            start_id = 74
            end_id = 100
        elif dataset == 'contract_rand_v0_1':
            start_id = 0
            end_id = 248
        elif dataset == 'qasper_rand_v1':
            start_id = 290
            end_id = 500

        toc_gen_model = "gpt-5.4"

        with open(Path(DATA_ROOT_FOLDER) / dataset / "queries.json", 'r') as f:
            queries = json.load(f)

        for qinfo in queries[start_id:end_id]:
            sht_gen_per_query(dataset, qinfo, toc_gen_model)


# if __name__ == "__main__":
#     # for dataset in DATASET_LIST[3:4]:
#     for dataset in ['finance']:
#         queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
#         with open(queries_path, 'r') as f:
#             queries = json.load(f)
        
#         num_queries = int(len(queries) * 0.2)
        
#         if dataset == "civic":
#             start_id = 0
#             end_id = len(queries)
#         elif dataset == "finance":
#             start_id = 0
#             end_id = 74
#         else:
#             start_id = 0
#             end_id = num_queries

#         sampled_docs = sorted(list(set([qinfo['file_name'] for qinfo in queries[start_id:end_id]])))
#         toc_gen_model = "gpt-5.4"

#         # pdf_folder = os.path.join(DATA_ROOT_FOLDER, dataset,  "pdf")

#         for filename in sampled_docs[:1]:
#             pdf_path = str(Path(DATA_ROOT_FOLDER) / dataset / "pdf" / f"{filename}.pdf")
#             sht_gen(dataset, pdf_path, toc_gen_model)





        
