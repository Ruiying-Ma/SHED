import fitz  # PyMuPDF
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config
import json
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder

def extract_toc_with_bbox(pdf_path):
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
        toc = doc.get_toc()  # [level, title, page number]
    except Exception as e:
        logging.warning(f"Failed to open or extract TOC from {pdf_path}: {e}")
        return None

    # metadata = doc.metadata
    # pdf_title = metadata.get("title", None)
    # if not pdf_title:
    #     logging.warning("PDF title metadata is missing.")
    #     return
    
    heading_identification = []
    node_clustering = []
    m_eid_level = {
        eid: toc[eid][0]
        for eid in range(len(toc))
    }

    for eid, entry in enumerate(toc):
        level, title, page_num = entry
        page = doc[page_num - 1]  # PyMuPDF pages are 0-indexed
        
        # Search for the heading text on the page
        text_instances = page.search_for(title)
        
        if text_instances:
            # If there are multiple instances, take the one whose text has the maximal number of capitalized letters
            # bbox = max(
            #     text_instances,
            #     key=lambda rect: sum(1 for c in page.get_textbox(rect) if c.isupper())
            # )
            bbox = text_instances[0]  # take the first occurrence
        else:
            bbox = None  # Heading not found on the page
            logging.warning(f"Heading '{title}' not found on page {page_num}.")
            return None

        # get the page_width and page_height
        page_width = page.rect.width
        page_height = page.rect.height

        # set left = left edge of the page
        # set top = top of the bounding box
        # width = new width of the bounding box when left margin of the bbox is expanded to the left edge of the page
        # height = height of the bounding box
        assert bbox is not None, "bbox should not be None here"
        left = 0
        top = bbox.y0
        # width = bbox.x1 - left
        width = page_width
        height = bbox.y1 - bbox.y0

        # extract new text from the new bounding box
        new_bbox = fitz.Rect(left, top, left + width, top + height)
        extracted_text = page.get_textbox(new_bbox).strip()
        new_header_entry = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": page_width,
            "page_height": page_height,
            "text": extracted_text,
            "type": "Section header",
        } # true headers
        assert eid == len(heading_identification), "eid should match heading_identification length: {} vs {}".format(eid, len(heading_identification))
        assert eid == len(node_clustering), "eid should match node_clustering length"
        parent_id = max(
            [i for i in m_eid_level if m_eid_level[i] < level and i < eid] or [-1]
        )
        new_cluster_entry = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": page_width,
            "page_height": page_height,
            "text": extracted_text.replace("\n", " "),
            "type": "Section header",
            "features": {},
            "id": eid, # node id, 0-based
            "cluster_id": parent_id,# parent_id = cluster_id
        } # true clusters
        heading_identification.append(new_header_entry)
        node_clustering.append(new_cluster_entry)
    
    

    # heading_identification_path = os.path.join(root_path, "intrinsic", "heading_identification", f"{filename}.json")
    # node_clustering_path = os.path.join(root_path, "intrinsic", "node_clustering", f"{filename}.json")
    os.makedirs(os.path.dirname(heading_identification_path), exist_ok=True)
    os.makedirs(os.path.dirname(node_clustering_path), exist_ok=True)
    assert not os.path.exists(heading_identification_path), f"heading_identification file already exists: {heading_identification_path}"
    assert not os.path.exists(node_clustering_path), f"node_clustering file already exists: {node_clustering_path}"
    with open(heading_identification_path, "w") as f:
        json.dump(heading_identification, f, indent=4)
    with open(node_clustering_path, "w") as f:
        json.dump(node_clustering, f, indent=4)

    return node_clustering_path


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




if __name__ == "__main__":
    pdf_dir = "/home/ruiying/SHTRAG/data/gov/pdf"
    for pdf_file in sorted(os.listdir(pdf_dir)):
        if not pdf_file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Processing {pdf_path}...")
        node_clustering_path = extract_toc_with_bbox(pdf_path)
        if node_clustering_path is not None:
            build_sht(node_clustering_path)
