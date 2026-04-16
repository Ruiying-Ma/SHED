import os
import sys
sys.path.append("/home/ruiying/SHTRAG/script_python")
from write_to_file import write_to_file

NEED_FAST = ''' -F "fast=true"'''

pdf_folder = "/home/ruiying/SHTRAG/data/finance/pdf"
heading_idenitification_folder = "/home/ruiying/SHTRAG/data/finance/heading_identification"
assert os.path.exists(heading_idenitification_folder)

bash_file = "/home/ruiying/SHTRAG/script_bash/financebench/dla.sh"
write_to_file(
    dest_path=bash_file,
    contents="",
    is_append=False,
    is_json=False,
)

for pdf_file in os.listdir(pdf_folder):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    assert os.path.exists(pdf_path)
    assert pdf_file.endswith(".pdf")
    output_path = os.path.join(heading_idenitification_folder, pdf_file.replace(".pdf", ".json"))
    assert not os.path.exists(output_path)
    command = f'''curl -X POST -F 'file=@{pdf_path}'{NEED_FAST} localhost:5060 | python -m json.tool > {output_path}'''

    write_to_file(
        dest_path=bash_file,
        contents=command + "\n",
        is_append=True,
        is_json=False,
    )

