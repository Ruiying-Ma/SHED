import os
import json
import re
from typing import List, Dict, Tuple
import subprocess
from datasets import load_dataset
# from rapidfuzz import fuzz

# # -------------------------
# # Header parsing (MD + RFC)
# # -------------------------

# MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# RFC_HEADER_RE = re.compile(
#     r"^((?:\d+(\.\d+)*)|Appendix\s+[A-Z])\.?\s+(.*)$"
# )

# def extract_headers_from_text(text: str) -> List[Dict]:
#     """
#     Extract headers from Markdown or RFC-style text.
#     Returns list of:
#       { "text": str, "level": int }
#     """
#     headers = []

#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue

#         # Markdown headers
#         m = MD_HEADER_RE.match(line)
#         if m:
#             level = len(m.group(1))
#             headers.append({
#                 "text": m.group(2).strip(),
#                 "level": level,
#             })
#             continue

#         # RFC headers
#         m = RFC_HEADER_RE.match(line)
#         if m:
#             numbering = m.group(1)
#             title = m.group(3).strip()

#             if numbering.startswith("Appendix"):
#                 level = 1
#             else:
#                 level = numbering.count(".") + 1

#             headers.append({
#                 "text": title,
#                 "level": level,
#             })

#     return headers



from weasyprint import HTML

HTML("example.html").write_pdf(
    "example.pdf",
    stylesheets=None
)

def markdown_to_html(md_path, pdf_path):
    """
    Convert markdown to PDF using pandoc.
    """
    cmd = [
        "pandoc",
        md_path,
        "-o",
        pdf_path
    ]
    subprocess.run(cmd, check=True)



def process_dataset(
    split: str = "train"
):

    ds = load_dataset(
        "meowterspace42/github-ai-project-docs",
        split=split
    )

    for i, ex in enumerate(ds):
        md_text = ex.get("text", "")
        if not md_text:
            continue

        metadata = ex['metadata']
        src_repo = metadata['source']
        norm_repo = src_repo.replace("/", "_").lower()
        md_path = f"/home/ruiying/SHTRAG/data/github/md/{norm_repo}.md"
        assert not os.path.exists(md_path), f"md file already exists: {md_path}"
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w") as f:
            f.write(md_text)
        pdf_path = f"/home/ruiying/SHTRAG/data/github/pdf/{norm_repo}.pdf"
        assert not os.path.exists(pdf_path), f"pdf file already exists: {pdf_path}"
        markdown_to_pdf(md_path, pdf_path)

if __name__ == "__main__":
    process_dataset(
        split="train"
    )
