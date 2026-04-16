import os
import tiktoken
import fitz
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging_config

pdf_folder = "/home/ruiying/SHTRAG/data/ccmain/pdf"
tokenizer = tiktoken.get_encoding("cl100k_base")

def list_pdfs(folder):
    pdf_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            pdf_files.append(filename)

    sorted_pdfs = sorted(pdf_files)

    print(sorted_pdfs[:10])

def record_pdf_metadat(folder):
    dst_csv_path = "/home/ruiying/SHTRAG/data/ccmain/ccmain_pdf_meta.csv"
    pdf_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            zip_id = int(filename.split("_")[0])
            pdf_id = int(filename.split("_")[1].replace(".pdf", "")) - 1
            assert zip_id >= 0
            assert pdf_id >= 0
            file_path = os.path.join(folder, filename)

            pdf_files.append((zip_id, pdf_id, file_path))

    sorted_pdfs = sorted(pdf_files, key=lambda x: (x[0], x[1]))

    with open(dst_csv_path, "a") as f:
        f.write("zip_id,pdf_id,page_num,token_num\n")



    for (zip_id, pdf_id, file_path) in sorted_pdfs:
        print(f"Processing {file_path}...")
        try:
            doc = fitz.open(file_path)
            page_num = doc.page_count
            total_tokens = 0
            for page in doc:
                text = page.get_text()
                tokens = tokenizer.encode(text)
                total_tokens += len(tokens)
            doc.close()

            with open(dst_csv_path, "a") as f:
                f.write(f"{zip_id},{pdf_id},{page_num},{total_tokens}\n")
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {e}")
    


if __name__ == "__main__":
    # list_pdfs(pdf_folder)
    record_pdf_metadat(pdf_folder)