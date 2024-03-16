"""
Script to extract text and tables from a pdf report.

The extracted data is saved to a json file.

This script uses the AzureConverter from the haystack library, which is a wrapper
for the Azure AI Document Intelligence API. The AzureConverter is used to extract
the text and table data from a pdf report and save it to a json file.
"""

import os
import argparse
from pathlib import Path

from haystack.nodes import AzureConverter
from dotenv import load_dotenv
from loguru import logger

from esg_retriever.config import PDF_REPORTS_DIR, JSON_REPORTS_DIR


load_dotenv()

AZURE_CONVERTER_KEY = os.environ.get("AZURE_CONVERTER_KEY")
"""Azure API key for the AzureConverter."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract text and tables from a pdf report and save it to a "
                    "json file."
        )

    parser.add_argument(
        "--pdf_filename",
        type=str,
        required=True,
        help=(
            f"File name of the pdf report to extract text and tables from. "
            f"e.g. 'ssw-IR22.pdf'. The pdf report is expected to be saved in the "
            f"directory `data/reports`.",
        ),
    )

    args = parser.parse_args()

    return args


def convert_report(pdf_filename: str):
    """
    Extract text and table data from a pdf report and save it to a json file.

    Uses Microsoft Azure AI Document Intelligence to extract the data.

    Parameters
    ----------
    pdf_filename : str
        File name of the pdf report to convert. The pdf report is expected to be saved
        in the directory {PDF_REPORTS_DIR}.
    """
    pdf_path = Path(PDF_REPORTS_DIR) / pdf_filename
    output_path = Path(JSON_REPORTS_DIR) / (pdf_path.stem + ".json")

    # Check if the pdf file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"Pdf report not found: {pdf_path}")

    # Check if the output file already exists
    if output_path.exists():
        logger.info(f"Skipping extraction of text and table data from {pdf_path} as "
                    f"output already exists: {output_path}")
        return

    converter = AzureConverter(
            endpoint="https://azureconverter.cognitiveservices.azure.com/",
            credential_key=AZURE_CONVERTER_KEY,
            model_id="prebuilt-layout",
            save_json=True,
        )

    # Extract the text and table data from the pdf
    logger.info(f"Extracting text and table data from {pdf_path}")
    try:
        converter.convert(file_path=pdf_path)
    except Exception as exc:
        logger.error(f"Failed to convert {pdf_path} to json: {exc}")
        return

    # Move the output json file to the JSON_REPORTS_DIR directory
    try:
        os.rename(pdf_path.with_suffix(".json"), output_path)
    except Exception as exc:
        logger.error(f"Failed to move {pdf_path} to {output_path}: {exc}")

    logger.info(f"Wrote extracted text and table data to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    pdf_filename = args.pdf_filename
    convert_report(pdf_filename)
