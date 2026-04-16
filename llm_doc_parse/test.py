# https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/formrecognizer/azure-ai-formrecognizer/samples/v3.2_and_later/sample_analyze_layout.py

# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: sample_analyze_layout.py

DESCRIPTION:
    This sample demonstrates how to extract text, selection marks, and layout information from a document
    given through a file.

    Note that selection marks returned from begin_analyze_document(model_id="prebuilt-layout") do not return the text
    associated with the checkbox. For the API to return this information, build a custom model to analyze the
    checkbox and its text. See sample_build_model.py for more information.

USAGE:
    python sample_analyze_layout.py

    Set the environment variables with your own values before running the sample:
    1) AZURE_FORM_RECOGNIZER_ENDPOINT - the endpoint to your Form Recognizer resource.
    2) AZURE_FORM_RECOGNIZER_KEY - your Form Recognizer API key
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import json
from azure.core.exceptions import HttpResponseError
import logging
import logging_config
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, AnalyzeResult

def analyze_documents_output_in_markdown(pdf_path):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    

    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

    logging.info("Creating Document Intelligence client...")
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    logging.info(f"Analyzing document {os.path.basename(pdf_path)}...")
    with open(pdf_path, 'rb') as pdf_file:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", document=pdf_file
        )

    logging.info("Waiting for analysis result...")
    result: AnalyzeResult = poller.result()

    logging.info("Analysis complete. Saving results to output.json and output.md...")
    result_dict = result.to_dict()

    with open("output.json", "w") as json_file:
        json.dump(result_dict, json_file, indent=4)

    with open("output.md", "w") as md_file:
        md_file.write(result.content)
    


if __name__ == "__main__":
    pdf_path = "/home/ruiying/SHTRAG/data/civic/pdf/01272021-1626.pdf"
    try:
        analyze_documents_output_in_markdown(pdf_path)
    except HttpResponseError as error:
        print(
            "For more information about troubleshooting errors, see the following guide: "
            "https://aka.ms/azsdk/python/formrecognizer/troubleshooting"
        )
        # Examples of how to check an HttpResponseError
        # Check by error code:
        if error.error is not None:
            if error.error.code == "InvalidImage":
                print(f"Received an invalid image error: {error.error}")
            if error.error.code == "InvalidRequest":
                print(f"Received an invalid request error: {error.error}")
            # Raise the error again after printing it
            raise
        # If the inner error is None and then it is possible to check the message to get more information:
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Uh-oh! Seems there was an invalid request: {error}")
        # Raise the error again
        raise