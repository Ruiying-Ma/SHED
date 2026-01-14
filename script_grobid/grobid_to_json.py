# from bs4 import BeautifulSoup
# from grobid2json import convert_xml_to_json
# import os

# xml_folder = "/home/ruiying/SHTRAG/data/finance/grobid/grobid.tei.xml"
# output_folder = "/home/ruiying/SHTRAG/data/finance/grobid/grobid"
# for xml_file in sorted(os.listdir(xml_folder))[:1]:
#     xml_path = os.path.join(xml_folder, xml_file)
#     assert os.path.isfile(xml_path)
#     file_path = xml_path
#     with open(file_path, "rb") as f:
#         xml_data = f.read()
#     soup = BeautifulSoup(xml_data, "xml")
#     paper_id = file_path.split("/")[-1].split(".")[0]
#     paper = convert_xml_to_json(soup, paper_id, "")
#     json_data = paper.as_json()
#     output_path = os.path.join(output_folder, paper_id + ".json")
#     with open(output_path, "w") as f:
#         f.write(json_data, indent=4)
    

import os
from grobid.tei import Parser
import json

xml_folder = "/home/ruiying/SHTRAG/data/finance/grobid/grobid.tei.xml"
output_folder = "/home/ruiying/SHTRAG/data/finance/grobid/grobid"
for xml_file in sorted(os.listdir(xml_folder))[:1]:
    xml_path = os.path.join(xml_folder, xml_file)
    assert os.path.isfile(xml_path)
    file_path = xml_path
    with open(file_path, "rb") as f:
        xml_content = f.read()

    parser = Parser(xml_content)
    article = parser.parse()
    json_data = article.to_json()  # raises RuntimeError if extra require 'json' not installed
    output_path = os.path.join(output_folder, xml_file.replace(".tei.xml", ".json"))
    with open(output_path, "w") as f:
        json.dump(json.loads(json_data), f, indent=4)
    print(f"Converted {xml_file} to JSON and saved to {output_path}")