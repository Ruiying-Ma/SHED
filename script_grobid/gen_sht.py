# Reference: https://github.com/kermitt2/grobid_client_python

import os
from grobid_client.grobid_client import GrobidClient

# Initialize with default localhost server
client = GrobidClient()

# Process documents
pdf_folder = "/home/ruiying/SHTRAG/data/finance/pdf"
output_folder = "/home/ruiying/SHTRAG/data/finance/grobid/grobid"
client.process(
    service="processFulltextDocument",
    input_path=pdf_folder,
    output=output_folder,
    n=10 # cocurrency level
)