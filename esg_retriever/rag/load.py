"""Module containing functions to load parsed pdfs."""

import os
from pathlib import Path

from haystack import Document
from haystack.nodes import AzureConverter
from loguru import logger

from dev.mapping import COMPANY_YEAR_PDF_MAPPING

AZURE_CONVERTER_KEY = os.environ.get("AZURE_CONVERTER_KEY")


AZURE_CONVERTER_DIR = "/home/tomw/unifi-pdf-llm/data/azureconverter_outputs"
"""Path to directory with json outputs from AzureConverter."""

TRAIN_CSV_PATH = "/home/tomw/unifi-pdf-llm/data/Train.csv"
"""Path to the Train.csv file."""


def load_documents(company: str, year: int) -> list[Document]:
    """
    Load documents for a company and year.

    Requires the corresponding pdf file(s) to have been previously converted to json
    using the AzureConverter.

    Parameters
    ----------
    company : str
        The company to load documents for.

    year : int
        The year to load documents for.

    Returns
    -------
    company_docs : list[Document]
        The documents for the company and year.

    Raises
    ------
    ValueError
        If no documents are found for the company and year.
    """
    company_docs = []
    converter = AzureConverter(
        endpoint="https://azureconverter.cognitiveservices.azure.com/",
        credential_key=AZURE_CONVERTER_KEY,
        model_id="prebuilt-layout",
    )

    try:
        file_name_list = COMPANY_YEAR_PDF_MAPPING[company][year]
    except KeyError:
        raise ValueError(f"No documents found for {company} in {year}")

    for file_name in file_name_list:
        file_name = file_name.replace(".pdf", ".json")
        file_path = Path(AZURE_CONVERTER_DIR) / file_name
        logger.info(f"Loading documents from {file_path}")
        docs = converter.convert_azure_json(file_path=file_path)
        company_docs.extend(docs)

    return company_docs
