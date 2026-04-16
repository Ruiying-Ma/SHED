from lxml import etree

xml_file = "sample.xml"
tree = etree.parse(xml_file)

toc = []

def extract_sections(sections, level=1):
    for sec in sections:
        title_el = sec.find("title")
        if title_el is not None:
            toc.append({"text": title_el.text, "level": level})
        # Recursively check for nested <sec>
        nested_secs = sec.findall("sec")
        if nested_secs:
            extract_sections(nested_secs, level=level+1)

body = tree.find("body")
sections = body.findall("sec")
extract_sections(sections)

# toc now contains all section titles and hierarchy levels
print(toc)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

pdf_file = "sample.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter
y = height - 50  # start from top margin

for item in toc:
    indent = 20 * (item['level'] - 1)
    c.setFont("Helvetica-Bold", 12 if item['level']==1 else 10)
    c.drawString(50 + indent, y, item['text'])
    y -= 20
    if y < 50:
        c.showPage()
        y = height - 50

c.save()

import fitz  # PyMuPDF
import json

pdf_path = "sample.pdf"
doc = fitz.open(pdf_path)

header_positions = []

for page_number in range(len(doc)):
    page = doc[page_number]
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] == 0:  # text block
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]])
                for toc_item in toc:
                    if toc_item["text"] == line_text:
                        bbox = line["bbox"]  # left, top, right, bottom
                        left, top, right, bottom = bbox
                        header_positions.append({
                            "text": toc_item["text"],
                            "level": toc_item["level"],
                            "page_number": page_number + 1,
                            "page_id": page.number,
                            "left": left,
                            "top": top,
                            "width": right - left,
                            "height": bottom - top,
                            "page_width": page.rect.width,
                            "page_height": page.rect.height
                        })

# Save to JSON
with open("toc_positions.json", "w") as f:
    json.dump(header_positions, f, indent=2)

print("Done! JSON saved with header positions.")
