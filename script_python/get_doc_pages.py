import pymupdf
import os

def get_doc_pages(file_path):
    doc = pymupdf.open(file_path)
    return doc.page_count


if __name__ == "__main__":

    for dataset in ["civic", "contract", "qasper", 'finance']:
        tot_page_num = 0
        root_folder = f"/home/ruiying/SHTRAG/data/{dataset}/pdf"
        for filename in sorted(os.listdir(root_folder)):
            pdf_path = os.path.join(root_folder, filename)
            page_num = get_doc_pages(pdf_path)
            tot_page_num += page_num
        print(f"Dataset: {dataset}, Total Pages: {tot_page_num}")