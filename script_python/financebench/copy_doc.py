import shutil
from doc_in_query import get_doc_in_query
import os

doc_path_list = get_doc_in_query()

for doc_path in doc_path_list:
    dest_path = os.path.join("/home/ruiying/SHTRAG/data/finance/pdf", os.path.basename(doc_path))
    assert not os.path.exists(dest_path)
    assert os.path.exists(doc_path)
    shutil.copy(doc_path, dest_path)