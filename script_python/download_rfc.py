import os
import time
import requests
from pathlib import Path

# ==============================
# Configuration
# ==============================
START_RFC = 7300
END_RFC = 7400    # download RFC 1–1000
OUTPUT_DIR = Path(f"/home/ruiying/SHTRAG/data/rfc_{END_RFC}")
SLEEP_SECONDS = 0.3     # be polite to the server
TIMEOUT = 20

HTML_URL = "https://www.rfc-editor.org/rfc/rfc{num}.html"
PDF_URL = "https://www.rfc-editor.org/rfc/pdfrfc/rfc{num}.txt.pdf"

HEADERS = {
    "User-Agent": "rfc-downloader/1.0 (research use)"
}

# ==============================
# Helpers
# ==============================
def download_file(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and r.content:
            out_path.write_bytes(r.content)
            return True
        return False
    except requests.RequestException:
        return False


# ==============================
# Main
# ==============================
def main():
    html_dir = OUTPUT_DIR / "html"
    pdf_dir = OUTPUT_DIR / "pdf"

    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    success_html = 0
    success_pdf = 0

    for rfc_num in range(START_RFC, END_RFC + 1):
        print(f"RFC {rfc_num}...")

        html_path = html_dir / f"rfc{rfc_num}.html"
        pdf_path = pdf_dir / f"rfc{rfc_num}.pdf"

        if not html_path.exists():
            html_ok = download_file(
                HTML_URL.format(num=rfc_num),
                html_path
            )
            success_html += int(html_ok)
        
        if not html_ok:
            print(f"  HTML download failed for RFC {rfc_num}, skipping PDF download.")
            continue

        if not pdf_path.exists():
            pdf_ok = download_file(
                PDF_URL.format(num=rfc_num),
                pdf_path
            )
            success_pdf += int(pdf_ok)

        if not pdf_ok:
            print(f"  PDF download failed for RFC {rfc_num}.")
            os.remove(html_path)
            assert not os.path.exists(pdf_path)
            success_html -= 1

        time.sleep(SLEEP_SECONDS)

    print("\nDownload summary")
    print(f"HTML files downloaded: {success_html}")
    print(f"PDF files downloaded:  {success_pdf}")


if __name__ == "__main__":
    main()
