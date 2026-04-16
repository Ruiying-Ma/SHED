import re
import json
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from pathlib import Path
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder


H_TAG_RE = re.compile(r"^h[1-6]$")
SPAN_H_RE = re.compile(r"h([1-6])")


def extract_toc_from_html(html_path: Path):
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")

    toc = []
    current_page = 1

    # Walk DOM in document order
    for elem in soup.body.descendants:
        if not getattr(elem, "attrs", None):
            continue

        # -------------------------
        # Page number anchors
        # -------------------------
        if "id" in elem.attrs:
            m = re.match(r"page-(\d+)", elem["id"])
            if m:
                current_page = int(m.group(1))
                logging.info(f"current_page={current_page}")

        # -------------------------
        # Case 1: real <h1>-<h6>
        # -------------------------
        if elem.name and H_TAG_RE.match(elem.name):
            text = elem.get_text(" ", strip=True)
            level = int(elem.name[1])

            toc.append({
                "text": text,
                "level": level,
                "page_number": current_page
            })
            logging.info(f"\theader={text}")
            continue

        # -------------------------
        # Case 2: <span class="hN">
        # -------------------------
        if elem.name == "span" and "class" in elem.attrs:
            h_class = next(
                (c for c in elem["class"] if SPAN_H_RE.fullmatch(c)),
                None
            )
            if not h_class:
                continue

            level = int(h_class[1])
            text = elem.get_text(" ", strip=True)

            toc.append({
                "text": text,
                "level": level,
                "page_number": current_page
            })
            logging.info(f"\theader={text}")

    return toc

def build_sht(node_clustering_path):
    root_dir = os.path.dirname(os.path.dirname(node_clustering_path))
    sht_skeleton_dir = os.path.join(root_dir, "sbert.gpt-4o-mini.c100.s100", "sht_skeleton")
    sht_vis_dir = os.path.join(root_dir, "sbert.gpt-4o-mini.c100.s100", "sht_vis")
    os.makedirs(sht_skeleton_dir, exist_ok=True)
    os.makedirs(sht_vis_dir, exist_ok=True)
    json_name = os.path.basename(node_clustering_path)

    # build_tree_skeleton
    if os.path.exists(os.path.join(sht_skeleton_dir, json_name)):
        logging.warning(f"SHT skeleton for {json_name} is already existed.")
        return
    print("Building SHT skeleton for", json_name)
    assert json_name.endswith(".json")
    assert os.path.exists(node_clustering_path), node_clustering_path
    assert not os.path.exists(os.path.join(sht_skeleton_dir, json_name)), f"SHT skeleton file already exists: {os.path.join(sht_skeleton_dir, json_name)}"
    assert not os.path.exists(os.path.join(sht_vis_dir, json_name.replace(".json", ".vis"))), f"SHT vis file already exists: {os.path.join(sht_vis_dir, json_name.replace('.json', '.vis'))}"
    with open(node_clustering_path, 'r') as file:
        objects = json.load(file)
    sht_builder = SHTBuilder(
        config=SHTBuilderConfig(
            store_json=os.path.join(sht_skeleton_dir, json_name),
            load_json=None,
            chunk_size=100,
            summary_len=100,
            embedding_model_name="sbert",
            summarization_model_name="gpt-4o-mini",
        )
    )
    logging.info(f"Built SHT skeleton for {json_name}")
    sht_builder.build(objects, 'wide')
    sht_builder.check()
    sht_builder.store2json()
    sht_builder.visualize(vis_path=os.path.join(sht_vis_dir, json_name.replace(".json", ".vis")))



# ------------------------------------
# PDF → geometry only
# ------------------------------------
def attach_pdf_geometry(pdf_path: Path, toc_struct):
    filename = os.path.basename(pdf_path).replace(".pdf", "")
    root_path = os.path.dirname(os.path.dirname(pdf_path))
    heading_identification_path = os.path.join(root_path, "intrinsic", "heading_identification", f"{filename}.json")
    node_clustering_path = os.path.join(root_path, "intrinsic", "node_clustering", f"{filename}.json")
    if os.path.exists(node_clustering_path):
        assert os.path.exists(heading_identification_path), "heading_identification should also exist if node_clustering exists"
        logging.warning(f"node clustering for {pdf_path} are already existed.")
        return node_clustering_path
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open PDF {pdf_path}: {e}")
        return None
    
    def clean_text(text: str) -> str:
        # Replace patterns like "1 . Title" or "12 . Title" with "1. Title"
        # return re.sub(r'(\d+)\s*\.\s*', r'\1. ', text)
        return '. '.join(text.split(' . ')).strip()

    heading_identification = []
    node_clustering = []
    m_eid_level = {
        toc_idx: toc_struct[toc_idx]["level"]
        for toc_idx in range(len(toc_struct))
    }

    for eid, entry in enumerate(toc_struct):
        level = entry["level"]
        title = clean_text(entry["text"])
        logging.info(f"Processing heading: {entry['text']}")
        page_num = entry["page_number"]
        page = doc[page_num - 1] # PyMuPDF pages are 0-indexed

        # Search for the heading text on the page
        text_instances = page.search_for(title)

        if text_instances:
            # Take the first instance
            bbox = text_instances[0]
        else:
            bbox = None
            print(f"Heading '{title}' not found on page {page_num}.")
            return None
        
        page_width = page.rect.width
        page_height = page.rect.height
        assert bbox is not None, "bbox should not be None here"
        left = bbox.x0
        top = bbox.y0
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0
        extracted_text = page.get_textbox(bbox).strip()
        new_header_entry = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": int(page_width),
            "page_height": int(page_height),
            "text": extracted_text,
            "type": "Section header",
        }
        assert eid == len(heading_identification) == len(node_clustering)
        parent_id = max(
            [i for i in m_eid_level if m_eid_level[i] < level and i < eid] or [-1]
        )
        new_cluster_entry = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": int(page_width),
            "page_height": int(page_height),
            "text": extracted_text,
            "type": "Section header",
            "features": {},
            "id": eid, # node id, 0-based
            "cluster_id": parent_id,# parent_id = cluster_id
        } # true clusters
        heading_identification.append(new_header_entry)
        node_clustering.append(new_cluster_entry)

    os.makedirs(os.path.dirname(heading_identification_path), exist_ok=True)
    os.makedirs(os.path.dirname(node_clustering_path), exist_ok=True)
    assert not os.path.exists(heading_identification_path), f"heading_identification file already exists: {heading_identification_path}"
    assert not os.path.exists(node_clustering_path), f"node_clustering file already exists: {node_clustering_path}"
    with open(heading_identification_path, "w") as f:
        json.dump(heading_identification, f, indent=4)
    with open(node_clustering_path, "w") as f:
        json.dump(node_clustering, f, indent=4)

    return node_clustering_path
    

# ------------------------------------
# Entry point
# ------------------------------------
if __name__ == "__main__":
    root_dir = "/home/ruiying/SHTRAG/data/rfc_7400"
    for html_name in os.listdir(os.path.join(root_dir, "html")):
        if not html_name.endswith(".html"):
            continue
        rfc_num = html_name.replace("rfc", "").replace(".html", "")
        html_path = Path(os.path.join(root_dir, "html", html_name))
        pdf_path = Path(os.path.join(root_dir, "pdf", f"rfc{rfc_num}.pdf"))

        toc_struct = extract_toc_from_html(html_path)
        node_clutsering_path = attach_pdf_geometry(pdf_path, toc_struct)
        if node_clutsering_path:
            build_sht(node_clutsering_path)
