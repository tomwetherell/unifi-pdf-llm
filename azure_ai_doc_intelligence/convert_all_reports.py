"""Script to convert all source pdfs to json using AzureConverter."""
# TODO: This should call function in convert_report.py

import os

from pathlib import Path
from loguru import logger
from haystack.nodes import AzureConverter

from esg_retriever.config import PDF_REPORTS_DIR, JSON_REPORTS_DIR


AZURE_CONVERTER_KEY = os.environ.get("AZURE_CONVERTER_KEY")
"""Azure API key for the AzureConverter."""


def convert_all_pdfs():
    """
    Convert all pdfs in the source directory to json using AzureConverter.

    Saves the json outputs to the output directory.
    """
    converter = AzureConverter(
        endpoint="https://azureconverter.cognitiveservices.azure.com/",
        credential_key=AZURE_CONVERTER_KEY,
        model_id="prebuilt-layout",  # Was "prebuilt-document"
        save_json=True,
    )

    for pdf_path in Path(PDF_REPORTS_DIR).rglob("*.pdf"):
        # Check if the output file already exists
        output_path = Path(JSON_REPORTS_DIR) / (pdf_path.stem + ".json")
        if output_path.exists():
            logger.info(f"Skipping {pdf_path} as output already exists")
            continue

        # Convert the pdf to json
        logger.info(f"Converting {pdf_path} to json")
        try:
            converter.convert(file_path=pdf_path)
        except Exception as exc:
            logger.error(f"Failed to convert {pdf_path} to json: {exc}")
            continue

        # Move the output file to the output directory
        try:
            os.rename(pdf_path.with_suffix(".json"), output_path)
        except Exception as exc:
            logger.error(f"Failed to move {pdf_path} to {output_path}: {exc}")

    logger.info("Finished")


if __name__ == "__main__":
    convert_all_pdfs()
