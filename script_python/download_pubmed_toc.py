import os
import json
import xml.etree.ElementTree as ET

input_folder = "/home/ruiying/SHTRAG/data/pubmed/xml"
output_folder = "/home/ruiying/SHTRAG/data/pubmed/intrinsic/sbert.gpt-4o-mini.c100.s100/sht_json"
os.makedirs(output_folder, exist_ok=True)

def parse_section(sec, level):
    """Recursively parse a Section node and its subsections."""
    toc = []
    # Get section title
    title_elem = sec.find("Title")
    if title_elem is not None and title_elem.text:
        toc.append({"text": title_elem.text.strip(), "level": level})
    
    # Look for nested sections
    for child_sec in sec.findall("Section"):
        toc.extend(parse_section(child_sec, level + 1))
    for child_sec in sec.findall("Sec"):  # PMC-style full-text
        toc.extend(parse_section(child_sec, level + 1))
    
    return toc

for xml_file in os.listdir(input_folder):
    if not xml_file.endswith(".xml"):
        continue

    path = os.path.join(input_folder, xml_file)
    tree = ET.parse(path)
    root = tree.getroot()

    # Start ToC list
    toc_list = []

    # Get PubMed ID
    pmid_elem = root.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

    # Add article title as level 0
    title_elem = root.find(".//ArticleTitle")
    if title_elem is not None and title_elem.text:
        toc_list.append({"text": title_elem.text.strip(), "level": 0})

    # Parse sections recursively (top-level)
    for sec in root.findall(".//Section") + root.findall(".//Sec"):
        toc_list.extend(parse_section(sec, level=1))

    # Optionally include abstract sections as level 1
    for abstract in root.findall(".//AbstractText"):
        if abstract.text:
            toc_list.append({"text": abstract.text.strip(), "level": 1})

    # Save JSON
    json_filename = os.path.join(output_folder, f"{pmid}_toc.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(toc_list, f, ensure_ascii=False, indent=2)

    print(f"Saved ToC for {pmid} -> {json_filename}")

print("Done! Hierarchical ToCs saved in JSON format.")
