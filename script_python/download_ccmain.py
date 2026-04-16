import requests
from bs4 import BeautifulSoup
import zipfile
import io
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz
import logging_config
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
from structured_rag.ClusteringOracle import ClusteringOracle, ClusteringOracleConfig
from eval.doc_class import is_local_first, is_global_first, HEADER_TYPES, is_well_formatted, is_loosely_formatted, is_depth_aligned
import json
import shutil

ROOT_DIR = "/home/ruiying/SHTRAG/data/ccmain"
os.makedirs(ROOT_DIR, exist_ok=True)
DST_CSV = os.path.join(ROOT_DIR, "ccmain_pdf_analysis.csv")
TMP_PDF = os.path.join(ROOT_DIR, "temp.pdf")
TMP_SHT_JSON = os.path.join(ROOT_DIR, "temp_sht.json")
TMP_SHT_VIS = os.path.join(ROOT_DIR, "temp_sht.vis")

PDF_DIR = os.path.join(ROOT_DIR, "pdf")
heading_identification_dir = os.path.join(ROOT_DIR, "intrinsic", "heading_identification")
node_clustering_dir = os.path.join(ROOT_DIR, "intrinsic", "node_clustering")
sht_dir = os.path.join(ROOT_DIR, "intrinsic", "sbert.gpt-4o-mini.c100.s100")
sht_skeleton_dir = os.path.join(sht_dir, "sht_skeleton")
sht_vis_dir = os.path.join(sht_dir, "sht_vis")
vp_dir = os.path.join(ROOT_DIR, "intrinsic", "visual_patterns")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(heading_identification_dir, exist_ok=True)
os.makedirs(node_clustering_dir, exist_ok=True)
os.makedirs(sht_skeleton_dir, exist_ok=True)
os.makedirs(sht_vis_dir, exist_ok=True)
os.makedirs(vp_dir, exist_ok=True)

def extract_toc_with_bbox(doc: fitz.Document, toc):
    heading_identification = []
    node_clustering = []
    m_eid_level = {
        eid: toc[eid][0]
        for eid in range(len(toc))
    }

    for eid, entry in enumerate(toc):
        level, title, page_num = entry
        if page_num <= 0:
            continue
        
        try:
            page = doc[page_num - 1]  # PyMuPDF pages are 0-indexed
            # Search for the heading text on the page
            text_instances = page.search_for(title)
        except Exception as e:
            return [], []
        
        if text_instances:
            # take the occurrence with the leftmost position
            bbox = min(
                text_instances,
                key=lambda rect: rect.x0
            )
        else:
            return [], []

        # get the page_width and page_height
        page_width = page.rect.width
        page_height = page.rect.height

        # set left = left edge of the page
        # set top = top of the bounding box
        # width = new width of the bounding box when left margin of the bbox is expanded to the left edge of the page
        # height = height of the bounding box
        assert bbox is not None, "bbox should not be None here"
        left = bbox.x0
        top = bbox.y0
        width = bbox.x1 - bbox.x0
        height = bbox.y1 - bbox.y0

        # extract new text from the new bounding box
        new_bbox = fitz.Rect(left, top, left + width, top + height)
        extracted_text = page.get_textbox(new_bbox).strip()
        new_header_entry = {
            "left": float(left),
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": int(page_width),
            "page_height": int(page_height),
            "text": extracted_text,
            "type": "Section header",
        } # true headers
        # assert eid == len(heading_identification), "eid should match heading_identification length: {} vs {}".format(eid, len(heading_identification))
        # assert eid == len(node_clustering), "eid should match node_clustering length"
        parent_id = max(
            [i for i in m_eid_level if m_eid_level[i] < level and i < eid] or [-1]
        )
        new_cluster_entry = {
            "left": float(left),
            "top": top,
            "width": width,
            "height": height,
            "page_number": page_num,
            "page_width": int(page_width),
            "page_height": int(page_height),
            "text": extracted_text.replace("\n", " "),
            "type": "Section header",
            "features": {},
            "id": len(node_clustering), # node id, 0-based
            "cluster_id": parent_id,# parent_id = cluster_id
        } # true clusters
        if new_header_entry["text"].strip() == "" or new_cluster_entry["text"].strip() == "":
            continue
        heading_identification.append(new_header_entry)
        node_clustering.append(new_cluster_entry)
    
    

    return heading_identification, node_clustering


def build_sht(heading_identification, node_clustering):
    if os.path.exists(TMP_SHT_JSON):
        os.remove(TMP_SHT_JSON)
    if os.path.exists(TMP_SHT_VIS):
        os.remove(TMP_SHT_VIS)
    objects = node_clustering
    sht_builder = SHTBuilder(
        config=SHTBuilderConfig(
            store_json=TMP_SHT_JSON,
            load_json=None,
            chunk_size=100,
            summary_len=100,
            embedding_model_name="sbert",
            summarization_model_name="gpt-4o-mini",
        )
    )
    sht_builder.build(objects, 'wide')
    sht_builder.check()
    sht_builder.store2json()
    sht_builder.visualize(vis_path=TMP_SHT_VIS)
    
    clustering_oracle = ClusteringOracle(ClusteringOracleConfig(store_json=None))
    try:
        visual_patterns = clustering_oracle.cluster(
            pdf_path=TMP_PDF,
            object_dicts_list=heading_identification
        )
    except Exception as e:
        return None, None # has_vp = False
    
    sht_nodes = sht_builder.tree['nodes']
    sht_headers = [n for n in sht_nodes if n['is_dummy'] == False and n['type'] in HEADER_TYPES]
    if len(sht_headers) != len(visual_patterns):
        return None, None
    return sht_headers, visual_patterns

def doc_class(sht_headers, visual_patterns):
    for h_node, v_pattern in zip(sht_headers, visual_patterns):
        h_node['inferred_cluster'] = v_pattern['cluster_id']
    
    is_wf = is_well_formatted(sht_headers)
    is_lf = is_loosely_formatted(sht_headers)
    is_da = is_depth_aligned(sht_headers)
    is_local = is_local_first(sht_headers)
    is_global = is_global_first(sht_headers)
    return is_wf, is_lf, is_da, is_local, is_global
            
        
    

def zip_url(zip_id: int) -> str:
    """
    Generate the download URL for a given ZIP ID.
    """
    base = "https://downloads.digitalcorpora.org/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles"
    subdir_start = (zip_id // 1000) * 1000
    subdir_end = subdir_start + 999
    return f"{base}/{subdir_start:04d}-{subdir_end:04d}/{zip_id:04d}.zip"


# def is_reachable_url(url: str, timeout=5) -> bool:
#     try:
#         r = requests.head(url, allow_redirects=True, timeout=timeout)
#         return r.status_code < 400
#     except requests.RequestException:
#         return False

def analyze_pdf_bytes(pdf_bytes: bytes):
    """
    Analyze a single PDF using fitz.
    """
    TMP_PDF = "temp.pdf"
    if os.path.exists(TMP_PDF):
        os.remove(TMP_PDF)
    try:
        with open(TMP_PDF, "wb") as f:
            f.write(pdf_bytes)

        doc = fitz.open(TMP_PDF)
        can_open = True
        has_toc = len(doc.get_toc()) > 0

        return can_open, has_toc

    except Exception:
        return False, False

    finally:
        if os.path.exists(TMP_PDF):
            os.remove(TMP_PDF)

def log_pdf_status(pfd_status):
    csv_line = ",".join([str(pfd_status[k]) for k in pfd_status]) + "\n"
    with open(DST_CSV, "a") as f:
        f.write(csv_line)

def process_zip(zip_id: int):
    url = zip_url(zip_id)

    try:
        r = requests.get(url, stream=True, timeout=(10, 1200))
        r.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"Failed to download ZIP ID {zip_id} from {url}: {e}")
        return False

    print(f"Processing ZIP ID {zip_id} from {url}")
    buffer = io.BytesIO()
    cid = 0
    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
        if cid % 1000 == 0:
            logging.debug(f"Downloading chunk {cid}...")
        buffer.write(chunk)
        cid += 1
    buffer.seek(0)

    pdf_cnt = 0
    with zipfile.ZipFile(buffer) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".pdf"):
                pdf_cnt += 1
                continue

            print(f"{zip_id}/{name}")
            pdf_status = {
                "zip_id": zip_id,
                "pdf_name": pdf_cnt,
                "can_open": False,
                "has_toc": False,
                "has_sht": False,
                "has_vp": False,
                "is_well_formatted": False,
                "is_loosely_formatted": False,
                "is_depth_aligned": False,
                "is_local_first": False,
                "is_global_first": False,
            }
            pdf_cnt += 1
            if os.path.exists(TMP_PDF):
                os.remove(TMP_PDF)
            if os.path.exists(TMP_SHT_JSON):
                os.remove(TMP_SHT_JSON)
            if os.path.exists(TMP_SHT_VIS):
                os.remove(TMP_SHT_VIS)

            try:
                pdf_bytes = z.read(name)
            except Exception as e:
                log_pdf_status(pdf_status)
                logging.warning(f"\tcan_open = False")
                continue

            
            if os.path.exists(TMP_PDF):
                os.remove(TMP_PDF)
            try:
                with open(TMP_PDF, "wb") as f:
                    f.write(pdf_bytes)
                doc = fitz.open(TMP_PDF)
            except Exception as e:
                log_pdf_status(pdf_status)
                logging.warning(f"\tcan_open = False")
                continue
                
            pdf_status["can_open"] = True

            try:
                toc = doc.get_toc()
            except Exception as e:
                log_pdf_status(pdf_status)
                logging.warning(f"\thas_toc = False")
                continue

            if len(toc) == 0:
                log_pdf_status(pdf_status)
                logging.warning(f"\thas_toc = False")
                continue

            pdf_status["has_toc"] = True
            heading_identification, node_clustering = extract_toc_with_bbox(doc, toc)
            if len(heading_identification) == 0 or len(node_clustering) == 0:
                log_pdf_status(pdf_status)
                logging.warning(f"\thas_sht = False")
                continue
            
            pdf_status["has_sht"] = True
            sht_headers, visual_patterns = build_sht(heading_identification, node_clustering)
            if sht_headers is None or visual_patterns is None:
                log_pdf_status(pdf_status)
                logging.warning(f"\thas_vp = False")
                continue
            pdf_status["has_vp"] = True

            # save
            pdf_path = os.path.join(PDF_DIR, f"{zip_id}_{pdf_cnt}.pdf")
            assert not os.path.exists(pdf_path), f"pdf_path {pdf_path} already exists"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            heading_identification_path = os.path.join(heading_identification_dir, f"{zip_id}_{pdf_cnt}.json")
            assert not os.path.exists(heading_identification_path), f"heading_identification_path {heading_identification_path} already exists"
            with open(heading_identification_path, "w") as f:
                json.dump(heading_identification, f, indent=4)
            node_clustering_path = os.path.join(node_clustering_dir, f"{zip_id}_{pdf_cnt}.json")
            assert not os.path.exists(node_clustering_path), f"node_clustering_path {node_clustering_path} already exists"
            with open(node_clustering_path, "w") as f:
                json.dump(node_clustering, f, indent=4)
            sht_json_path = os.path.join(sht_skeleton_dir, f"{zip_id}_{pdf_cnt}.json")
            assert not os.path.exists(sht_json_path), f"sht_json_path {sht_json_path} already exists"
            with open(sht_json_path, "w") as f:
                json.dump(sht_headers, f, indent=4)
            sht_vis_path = os.path.join(sht_vis_dir, f"{zip_id}_{pdf_cnt}.vis")
            assert not os.path.exists(sht_vis_path), f"sht_vis_path {sht_vis_path} already exists"
            if os.path.exists(TMP_SHT_VIS):
                shutil.copyfile(TMP_SHT_VIS, sht_vis_path)
            vp_path = os.path.join(vp_dir, f"{zip_id}_{pdf_cnt}.json")
            assert not os.path.exists(vp_path), f"vp_path {vp_path} already exists"
            with open(vp_path, "w") as f:
                json.dump(visual_patterns, f, indent=4)


            is_wf, is_lf, is_da, is_local, is_global = doc_class(sht_headers, visual_patterns)
            pdf_status["is_well_formatted"] = is_wf
            pdf_status["is_loosely_formatted"] = is_lf
            pdf_status["is_depth_aligned"] = is_da
            pdf_status["is_local_first"] = is_local
            pdf_status["is_global_first"] = is_global
            log_pdf_status(pdf_status)


if __name__ == "__main__":
    # with open(DST_CSV, "w") as f:
    #     f.write("zip_id,pdf_name,can_open,has_toc,has_sht,has_vp,is_well_formatted,is_loosely_formatted,is_depth_aligned,is_local_first,is_global_first\n")

    for zip_id in range(1189, 7932):
        process_zip(zip_id)





    # import json
    # pdf_status = {
    #             "can_open": False,
    #             "has_toc": False,
    #             "has_sht": False,
    #             "has_vp": False,
    #             "is_well_formatted": False,
    #             "is_loosely_formatted": False,
    #             "is_depth_aligned": False,
    #             "is_local_first": False,
    #             "is_global_first": False,
    #         }


    # try:
    #     doc = fitz.open(TMP_PDF)
    # except Exception as e:
    #     log_pdf_status(pdf_status)
    #     logging.warning(f"\tcan_open = False")
    #     exit(0)
        
    # pdf_status["can_open"] = True

    # try:
    #     toc = doc.get_toc()
    # except Exception as e:
    #     log_pdf_status(pdf_status)
    #     logging.warning(f"\thas_toc = False")
    #     exit(0)

    # if len(toc) == 0:
    #     log_pdf_status(pdf_status)
    #     logging.warning(f"\thas_toc = False")
    #     exit(0)

    # pdf_status["has_toc"] = True
    # heading_identification, node_clustering = extract_toc_with_bbox(doc, toc)
    # if len(heading_identification) == 0 or len(node_clustering) == 0:
    #     log_pdf_status(pdf_status)
    #     logging.warning(f"\thas_sht = False")
    #     exit(0)
    
    # pdf_status["has_sht"] = True
    # sht_headers, visual_patterns = build_sht(heading_identification, node_clustering)
    # if sht_headers is None or visual_patterns is None:
    #     log_pdf_status(pdf_status)
    #     logging.warning(f"\thas_vp = False")
    #     exit(0)

    # pdf_status["has_vp"] = True

    # with open(os.path.join(ROOT_DIR, "temp_vp.json"), "w") as f:
    #     json.dump(visual_patterns, f, indent=4)

    # is_wf, is_lf, is_da, is_local, is_global = doc_class(sht_headers, visual_patterns)
    # pdf_status["is_well_formatted"] = is_wf
    # pdf_status["is_loosely_formatted"] = is_lf
    # pdf_status["is_depth_aligned"] = is_da
    # pdf_status["is_local_first"] = is_local
    # pdf_status["is_global_first"] = is_global
    # log_pdf_status(pdf_status)