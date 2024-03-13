"""Module containing functions to load parsed pdfs."""
# TODO: Create load_and_preprocess_documents function

import os
from pathlib import Path

from haystack import Document
from haystack.nodes import AzureConverter
from dotenv import load_dotenv
from loguru import logger

from esg_retriever.preprocess import preprocess_documents
from esg_retriever.config import JSON_REPORTS_DIR, COMPANY_YEAR_PDF_MAPPING


load_dotenv()

AZURE_CONVERTER_KEY = os.environ.get("AZURE_CONVERTER_KEY")


def load_and_preprocess_documents(
    company: str,
    year: int,
    window_size: int = 5,
    discard_text: bool = True,
) -> list[Document]:
    """
    Load and preprocess documents for a company and year.

    Requires the corresponding pdf file(s) to have been previously converted to json
    using the AzureConverter.

    Parameters
    ----------
    company : str
        The company to load documents for.

    year : int
        The year to load documents for.

    window_size : int
        The size of the sliding window used to split the tables.

    discard_text : bool
        If True, discard text passages and keep only tables.

    Returns
    -------
    company_docs : list[Document]
        The preprocessed documents for the company and year.

    Raises
    ------
    ValueError
        If no documents are found for the company and year.
    """
    company_docs = load_documents(company, year)
    company_docs = preprocess_documents(company_docs, window_size, discard_text)

    return company_docs


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
        file_path = Path(JSON_REPORTS_DIR) / file_name
        logger.info(f"Loading documents from {file_path}")
        docs = converter.convert_azure_json(file_path=file_path)
        company_docs.extend(docs)

    return company_docs
