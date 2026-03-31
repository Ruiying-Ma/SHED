import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from script_python.finance_tables import reform_context
import config
from agents.get_doc_txt import get_doc_txt

if __name__ == "__main__":
    node_clustering_folder = "/home/ruiying/SHTRAG/data/finance/node_clustering"
    table_folder = "/home/ruiying/SHTRAG/data/finance/docling_tables"
    pdf_folder = "/home/ruiying/SHTRAG/data/finance/pdf"
    txt_folder = "/home/ruiying/SHTRAG/data/finance/doc_txt_reform_table"
    os.makedirs(txt_folder, exist_ok=True)
    for pdf_filename in os.listdir(pdf_folder):
        file_name = pdf_filename.replace(".pdf", "")
        print(f"Reforming context for file: {file_name}")
        doc_txt = get_doc_txt("finance", file_name)
        node_clustering_path = os.path.join(node_clustering_folder, f"{file_name}.json")
        print(f"\tLoading node clustering from: {node_clustering_path}")
        assert os.path.exists(node_clustering_path), node_clustering_path
        table_path = os.path.join(table_folder, f"{file_name}.json")
        print(f"\tLoading tables from: {table_path}")
        assert os.path.exists(table_path), table_path
        with open(node_clustering_path, "r") as f:
            node_clustering = json.load(f)
        with open(table_path, "r") as f:
            table_list = json.load(f)
        orig_context = doc_txt
        order = True
        new_context = reform_context(orig_context, node_clustering, table_list, order)
        new_context_path = os.path.join(txt_folder, f"{file_name}.txt")
        with open(new_context_path, "w") as f:
            f.write(new_context)