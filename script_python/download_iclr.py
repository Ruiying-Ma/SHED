import os
import requests
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config

# ================================
# Configuration
# ================================
# MAX_PAPERS = 10
YEAR = 2018  # ICLR conference year
OUTPUT_FOLDER = f"/home/ruiying/SHTRAG/data/iclr_{YEAR}/pdf"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# OpenReview API endpoint for accepted papers
API_URL = f"https://api.openreview.net/notes?invitation=ICLR.cc/{YEAR}/Conference/-/Blind_Submission"
BASE_URL = "https://openreview.net"


# ================================
# Fetch the paper list
# ================================
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
response = requests.get(API_URL, headers=headers)
if response.status_code != 200:
    raise Exception(f"Failed to fetch data: {response.status_code}")

data = response.json()
notes = data.get("notes", [])

logging.info(f"Found {len(notes)} papers for year {YEAR}.")
assert len(set([n['number'] for n in notes])) == len(notes), "Duplicate paper numbers found!"

# ================================
# Download each paper PDF
# ================================
for note in notes:
    id = str(note.get("number"))
    if not id:
        logging.warning(f"Skipping note with no ID: {note}")
        continue
    # title = note.get("content", {}).get("title", "untitled").replace("/", "_")
    pdf_url = note.get("content", {}).get("pdf")
    if not pdf_url:
        logging.warning(f"No PDF for paper: {id}")
        continue
    
    pdf_url = BASE_URL + pdf_url  # <-- Fix: prepend domain
    pdf_path = os.path.join(OUTPUT_FOLDER, f"{id}.pdf")
    assert not os.path.exists(pdf_path), f"File already exists: {pdf_path}"
    try:
        pdf_data = requests.get(pdf_url)
        with open(pdf_path, "wb") as f:
            f.write(pdf_data.content)
        # print(f"Downloaded: {id}")
    except Exception as e:
        logging.warning(f"Failed to download {id}: {e}")

