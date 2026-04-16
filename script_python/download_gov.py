import fitz  # PyMuPDF
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config
import json
from structured_rag import StructuredRAG, SHTBuilderConfig, SHTBuilder
import re
import shutil

def has_toc(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # [level, title, page number]
        assert len(toc) > 0
        logging.info(f"Extracted TOC from {pdf_path}")
        return True
    except Exception as e:
        # logging.warning(f"Failed to open or extract TOC from {pdf_path}: {e}")
        return False


if __name__ == "__main__":
    has_toc_cnt = 0
    pdf_dir = "/home/ruiying/SHTRAG/data/gov/raw_pdf"
    for pdf_file in sorted(os.listdir(pdf_dir)):
        if not pdf_file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, pdf_file)
        has_toc_flag = has_toc(pdf_path)
        if has_toc_flag:
            has_toc_cnt += 1
            shutil.copy(pdf_path, "/home/ruiying/SHTRAG/data/gov/pdf/")


