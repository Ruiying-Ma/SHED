# https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision?tabs=rest#use-a-local-image
# import module
from pdf2image import convert_from_path
from PIL import Image
import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from dotenv import load_dotenv
import math
import json
from datetime import datetime
import traceback
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import logging_config
import fitz

VISION_PARSE_PROMPT_TEMPL = '''Task:
Generate a detailed and structured Table of Contents (ToC) for the given document. The ToC must accurately capture **all headers** (including the document title, section headers, subsection headers, sub-subsection headers, etc.) and clearly represent their **hierarchical structure**.

Instructions:
1. Do not modify any headers.
   - Use the exact text of each header as it appears in the document.
2. Use the following Markdown format to show hierarchy:

   ```markdown
   # Title  
   ## Section Header  
   ### Subsection Header  
   #### Sub-subsection Header  
   ...
   ```

   - The number of `#` symbols indicates the hierarchy level.
   - Include one space between the `#` symbols and the header text.
   - Preserve the order in which the headers appear in the document.

3. Output only the Table of Contents, with nothing else.

The document is provided as images extracted from a PDF file. Each image corresponds to a page in the document. The images are ordered following the sequence of pages in the PDF.{note}

Your Table of Contents:
<your ToC here>'''

FULL_DOC = ""
BEGIN_DOC = " The given images only contain the beginning part of the document."
END_DOC = " The given images only contain the ending part of the document."
MID_DOC = " The given images contain a middle part of the document, omitting both the beginning and the end."

MAX_INPUT_TOKENS = 126000

SAFETY_CHECK = True


def _pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def _image_token(img: Image.Image):
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    
    w0, h0 = img.size
    # Step 1: fit within 2048x2048
    scale1 = min(2048 / w0, 2048 / h0, 1.0)  # only shrink if larger, do not upscale
    w1 = max(1, int(round(w0 * scale1)))
    h1 = max(1, int(round(h0 * scale1)))
    
    # Step 2: if shortest side > 768, scale so shortest == 768
    shortest = min(w1, h1)
    if shortest > 768:
        scale2 = 768 / shortest
        w2 = max(1, int(round(w1 * scale2)))
        h2 = max(1, int(round(h1 * scale2)))
    else:
        w2, h2 = w1, h1
    
    # Step 3: tile counts (512x512), rounding up partial tiles
    tiles_x = math.ceil(w2 / 512)
    tiles_y = math.ceil(h2 / 512)
    total_tiles = tiles_x * tiles_y
    
    # Step 4: token calculations
    per_tile_5667 = 5667
    base_5667 = 2833
    
    tokens_5667 = total_tiles * per_tile_5667 + base_5667
    return tokens_5667



# Function to encode a local image into data URL 
def _local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def vision_parse(pdf_path, result_path):
    # images = _pdf_to_images(pdf_path)
    # image_tokens = [_image_token(img) for img in images]
    # tot_tokens = sum(image_tokens)
    # group_num = math.ceil(tot_tokens / MAX_INPUT_TOKENS)
    # image_per_group = math.ceil(len(images) / group_num)

    existing_results = []
    if os.path.exists(result_path):
        with open(result_path, "r") as file:
            for l in file:
                existing_results.append(json.loads(l)["id"])

    # image_groups = []
    # for i in range(group_num):
    #     start_idx = i * image_per_group
    #     end_idx = min((i + 1) * image_per_group, len(images))
    #     if start_idx < end_idx:
    #         image_groups.append(images[start_idx:end_idx])
    # assert sum(len(g) for g in image_groups) == len(images)
    # assert sum(sum(_image_token(img) for img in g) for g in image_groups) == tot_tokens

    # tot_costs = tot_tokens * 0.15 / 1000000
    # print(f"\tpages: {len(images)}, groups: {len(image_groups)}, estimated cost: ${tot_costs:.6f}")


    pdf_page_num = len(fitz.open(pdf_path))
    image_groups = [None for _ in range(pdf_page_num)]

    assert VISION_PARSE_PROMPT_TEMPL.count("{note}") == 1
    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\t[{cur_time}] Start processing {len(image_groups)} groups...")
    for i, img_group in enumerate(image_groups):
        if i in existing_results:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]⚠️ Skipping already processed group {i}")
            continue
        prompt = None
        if i == 0 and i == len(image_groups) - 1:
            prompt = VISION_PARSE_PROMPT_TEMPL.replace("{note}", FULL_DOC)
        elif i == 0:
            prompt = VISION_PARSE_PROMPT_TEMPL.replace("{note}", BEGIN_DOC)
        elif i == len(image_groups) - 1:
            prompt = VISION_PARSE_PROMPT_TEMPL.replace("{note}", END_DOC)
        else:
            prompt = VISION_PARSE_PROMPT_TEMPL.replace("{note}", MID_DOC)

        assert prompt != None

        # Convert images to data URLs
        image_data_urls = []
        img_group = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
        for j, img in enumerate(img_group):
            img_path = f"tmp_img/{i}_{j}.jpg"
            assert not os.path.exists(img_path)
            img.save(img_path, 'JPEG')
            data_url = _local_image_to_data_url(img_path)
            image_data_urls.append(data_url)
            os.remove(img_path)

        # Initialize Azure OpenAI client
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ]

        for data_url in image_data_urls:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": "high"
                    }
                }
            )
        
        logging.info(f"message_len = {len(messages[0]['content'])}")

        if SAFETY_CHECK == False:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                with open(result_path, "a") as f:
                    answer_info = {
                        "id": i,
                        "headers": response.choices[0].message.content
                    }
                    f.write(json.dumps(answer_info) + "\n")
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]✅ Finished processing group {i}")
            except Exception as e:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]❌ Error processing group {i}: {e}\n{traceback.print_exc()}")


    # return tot_costs
        

if __name__ == "__main__":
    logging.disable(level=logging.DEBUG)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-check", action="store_true", help="Enable safety check")
    # parser.add_argument("--pdf-path", type=str, required=True, help="Path to the PDF file")

    args = parser.parse_args()
    SAFETY_CHECK = args.safety_check

    for dataset in ['finance']:
        tot_cost = 0
        root_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "pdf")
        for filename in sorted(os.listdir(root_folder)):
            if filename not in [
                "WALMART_2018_10K.pdf",
                "PEPSICO_2021_10K.pdf",
                "PEPSICO_2022_10K.pdf",
            ]:
                continue
            pdf_path = os.path.join(root_folder, filename)

            result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, 'llm_vision', 'llm_vision', os.path.basename(pdf_path).replace('.pdf', '.jsonl'))
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{cur_time}] {pdf_path}...")
            logging.info(f"Processing file: {filename}")
            try:
                per_cost = vision_parse(pdf_path, result_path)
                tot_cost += per_cost
            except Exception as e:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\t[{cur_time}]❌ Error processing {pdf_path}: {e}\n{traceback.print_exc()}")

        print(f"Dataset: {dataset}, Total Cost: ${tot_cost:.6f}")

