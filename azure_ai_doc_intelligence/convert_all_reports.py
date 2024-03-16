"""Script to convert all source pdfs to json using AzureConverter."""

import os


from convert_report import convert_report
from esg_retriever.config import PDF_REPORTS_DIR


AZURE_CONVERTER_KEY = os.environ.get("AZURE_CONVERTER_KEY")
"""Azure API key for the AzureConverter."""


def convert_all_pdfs():
    """
    Convert all pdfs in the source directory to json using AzureConverter.

    Saves the json outputs to the output directory.
    """
    # Loop over the pdf filenames in the source directory
    for pdf_filename in os.listdir(PDF_REPORTS_DIR):
        convert_report(pdf_filename)


if __name__ == "__main__":
    convert_all_pdfs()
