import os
import json
import requests
from tqdm import tqdm
from lxml import etree
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config

# ---------------- CONFIG ----------------

PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
ROOT_DIR = "/home/ruiying/SHTRAG/data/pubmed"
META_DIR = os.path.join(ROOT_DIR, "meta")
XML_DIR = os.path.join(ROOT_DIR, "xml")
PDF_DIR = os.path.join(ROOT_DIR, "pdf")
MAX_DOCS = 1

os.makedirs(XML_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
import xml.etree.ElementTree as ET
def fetch_pmc_ids(n=100):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": "open access[filter]",
        "retmax": n
    }
    r = requests.get(url, params=params)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    return [id_.text for id_ in root.findall(".//Id")]

from itertools import islice
import time

def uids_to_pmcids(uids, batch_size=200, sleep=0.34):
    """
    Convert Entrez PMC UIDs to human-readable PMCIDs.

    - Uses POST to avoid 414 errors
    - Batches requests
    - Respects NCBI rate limits
    """

    pmcids = []
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def batched(iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    for batch in batched(uids, batch_size):
        data = {
            "db": "pmc",
            "id": ",".join(batch),
            "retmode": "xml"
        }

        r = requests.post(url, data=data, timeout=30)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        pmcids = []
        for docsum in root.findall(".//DocSum"):
            for item in docsum.findall(".//Item"):
                if item.attrib.get("Name") == "pmcid":
                    pmcids.append(item.text)
        pmcids.extend(pmcids)
        time.sleep(sleep)  # NCBI politeness

    return pmcids



def download_xml(pmcid):
    # url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/?format=xml"
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/?page=xml"
    r = requests.get(url)
    if r.status_code != 200:
        logging.warning(f"Failed to download XML for {pmcid} from {url}")
        return None
    path = os.path.join(XML_DIR, f"{pmcid}.xml")
    with open(path, "wb") as f:
        f.write(r.content)
    logging.info(f"Downloaded XML for {pmcid}")
    return path


def extract_toc(xml_path):
    """Extract explicit ToC from <sec> hierarchy"""
    tree = etree.parse(xml_path)
    secs = tree.xpath("//body//sec")
    if not secs:
        logging.warning(f"No sections found in {xml_path}")
        return None

    toc = []

    def walk(sec, depth=0):
        title = sec.findtext("title")
        if title:
            entry = {
                "title": title.strip(),
                "depth": depth,
                "children": []
            }
            for child in sec.findall("sec"):
                child_entry = walk(child, depth + 1)
                if child_entry:
                    entry["children"].append(child_entry)
            return entry
        return None

    for sec in tree.xpath("//body/sec"):
        item = walk(sec, 0)
        if item:
            toc.append(item)

    return toc if toc else None


# ---------------- PDF RENDERING ----------------

class TrackingDoc(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        self.header_positions = []
        super().__init__(*args, **kwargs)

    def afterFlowable(self, flowable):
        # Only track objects marked as headers
        if hasattr(flowable, "_is_header"):
            # Width/height of flowable
            width, height = flowable.wrap(self.width, self.height)

            # Current page
            page = self.canv.getPageNumber()

            # Y position: you can get from the canvas's _y attribute
            y = self.canv._y  # this is protected, but exists
            x = self.leftMargin

            self.header_positions.append({
                "title": getattr(flowable, "text", ""),
                "page": page,
                "bbox": [x, y - height, x + width, y]  # y is top, so subtract height for bottom
            })


def xml_to_pdf(xml_path, pdf_path):
    tree = etree.parse(xml_path)
    styles = getSampleStyleSheet()

    doc = TrackingDoc(pdf_path, pagesize=LETTER)
    story = []

    for sec in tree.xpath("//body//sec"):
        title = sec.findtext("title")
        if title:
            p = Paragraph(f"<b>{title}</b>", styles["Heading2"])
            p._is_header = True
            p.text = title
            story.append(p)
            story.append(Spacer(1, 12))

        for p_text in sec.xpath(".//p//text()"):
            story.append(Paragraph(p_text, styles["BodyText"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    return doc.header_positions


# ---------------- MAIN PIPELINE ----------------


# collected = 0
# uids = fetch_pmc_ids(600)
# pmcids = uids_to_pmcids(uids)

# for pmcid in tqdm(pmcids):
#     if collected >= MAX_DOCS:
#         break

#     xml_path = download_xml(pmcid)
#     if not xml_path:
#         continue

#     toc = extract_toc(xml_path)
#     if toc is None:
#         logging.warning(f"No ToC extracted for {pmcid}, skipping.")
#         os.remove(xml_path)
#         continue

#     pdf_path = os.path.join(PDF_DIR, f"{pmcid}.pdf")
#     header_positions = xml_to_pdf(xml_path, pdf_path)

#     meta = {
#         "pmcid": pmcid,
#         "toc": toc,
#         "header_positions": header_positions
#     }

#     with open(os.path.join(META_DIR, f"{pmcid}.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     collected += 1
#     print(f"Collected {collected}/{MAX_DOCS}: {pmcid}")

# print("Done.")

xml_path = "sample.xml"
pmcid = "samples"
toc = extract_toc(xml_path)
if toc is None:
    logging.warning(f"No ToC extracted for {pmcid}, skipping.")
    os.remove(xml_path)

print(toc)

pdf_path = os.path.join(PDF_DIR, f"{pmcid}.pdf")
header_positions = xml_to_pdf(xml_path, pdf_path)

meta = {
    "pmcid": pmcid,
    "toc": toc,
    "header_positions": header_positions
}

with open(os.path.join(META_DIR, f"{pmcid}.json"), "w") as f:
    json.dump(meta, f, indent=2)


