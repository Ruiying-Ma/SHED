# https://developers.llamaindex.ai/python/cloud/llamaparse/features/multimodal/

# parser = LlamaParse(
#   use_vendor_multimodal_model=True,
#   azure_openai_deployment_name="llamaparse-gpt-4o",
#   azure_openai_endpoint="https://<org>.openai.azure.com/openai/deployments/<dep>/chat/completions?api-version=<ver>",
#   azure_openai_api_version="2024-02-15-preview",
#   azure_openai_key="xxx"
# )


from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import logging

def multimodal_parser(pdf_path):
    openai_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    assert os.path.exists(openai_key_path)
    load_dotenv(openai_key_path)

    parser = LlamaParse(
        use_vendor_multimodal_model=True,
        azure_openai_deployment_name="gpt-4o-mini",
        azure_openai_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_openai_api_version=os.getenv("AZURE_API_VERSION"),
        azure_openai_key=os.getenv("AZURE_API_KEY")
    )
    logging.info(f"Starting multimodal parsing {pdf_path}...")
    result = parser.parse(pdf_path)
    logging.info(f"Completed multimodal parsing {pdf_path}.")
    # get the llama-index markdown documents
    logging.info(f"\tExtracting markdown and text documents...")
    markdown_documents = result.get_markdown_documents(split_by_page=False)

    with open("tmp_md.md", "w") as f:
        for doc in markdown_documents:
            f.write(doc.get_content())
            f.write("\n\n---\n\n")

    # get the llama-index text documents
    logging.info(f"\tExtracting text documents...")
    text_documents = result.get_text_documents(split_by_page=False)

    with open("tmp_txt.txt", "w") as f:
        for doc in text_documents:
            f.write(doc.get_content())
            f.write("\n\n---\n\n")


    # access the raw job result
    # Items will vary based on the parser configuration
    logging.info(f"\tAccessing raw parsing result...")
    for page in result.pages:
        print(page.text)
        print(page.md)
        print(page.images)
        print(page.layout)
        print(page.structuredData)

    logging.info(f"\tSaving full parsing result to tmp.json...")
    with open("tmp.json", "w") as f:
        f.write(result.model_dump_json(indent=4))


if __name__ == "__main__":
    pdf_path = "/home/ruiying/SHTRAG/data/civic/pdf/01272021-1626.pdf"
    multimodal_parser(pdf_path)