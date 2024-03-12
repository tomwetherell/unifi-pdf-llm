"""Script to make submission for competition."""

from pathlib import Path

import pandas as pd

from esg_retriever.load import load_documents
from esg_retriever.preprocess import preprocess_documents
from esg_retriever.rag import ModularRAG
from esg_retriever.config import MISC_DATA_DIR, COMPANY_YEAR_PDF_MAPPING, COMPANIES


SUBMISSION_TEMPLATE_PATH = MISC_DATA_DIR / Path("SampleSubmission.csv")
"""Path to the sample submission template."""


def make_submission():
    """Make a submission for the competition."""
    submission_df = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    models = {}

    # Load the ModularRAG model for each company
    for company in COMPANIES:
        years = list(COMPANY_YEAR_PDF_MAPPING[company].keys())
        submission_year = max(years)
        docs = load_documents(company, submission_year)
        docs = preprocess_documents(docs)
        models[company] = ModularRAG(docs=docs, company=company)

    # TODO: Loop through the submission dataframe and make predictions
    pass


def _extract_amkey(amkey_company_id: str) -> str:
    """
    Return the AMKEY from the id.

    Parameters
    ----------
    amkey_company_id : str
        An ID in the format <AMKEY>_X_<COMPANY>. For example, "100_X_Absa".

    Returns
    -------
    amkey : str
        The AMKEY extracted from the id.
    """
    amkey = amkey_company_id.split("_")[0]
    return amkey


def _extract_company(amkey_company_id: str) -> str:
    """
    Return the company from the id.

    Parameters
    ----------
    amkey_company_id : str
        An ID in the format <AMKEY>_X_<COMPANY>. For example, "100_X_Absa".

    Returns
    -------
    company : str
        The company extracted from the id.
    """
    company = amkey_company_id.split("_")[-1]
    return company


if __name__ == "__main__":
    make_submission()
