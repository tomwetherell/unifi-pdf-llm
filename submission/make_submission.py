"""Script to make submission for competition."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from esg_retriever.load import load_and_preprocess_documents
from esg_retriever.rag import ModularRAG
from esg_retriever.utils import list_all_amkeys
from esg_retriever.config import COMPANY_YEAR_PDF_MAPPING, COMPANIES


SUBMISSION_LOG_FILE = Path(__file__).resolve().parent / "submission.log"
"""File to save the logs from making the submission."""


def retrieve_all_amkey_values(company: str, year: int) -> dict:
    """
    Return all AMKEY values for a company and year.

    Parameters
    ----------
    company : str
        The company name.

    year : int
        The year to retrieve the AMKEY values for.

    Returns
    -------
    amkey_values : dict[int, float | None]
        A dictionary with the AMKEY as the key and the value as the value.
    """
    amkey_values = {}
    amkeys = list_all_amkeys()

    docs = load_and_preprocess_documents(company, year, window_size=2, discard_text=True)
    rag = ModularRAG(docs=docs, company=company)

    for amkey in amkeys[:2]:
        value = rag.query(amkey, year=year)

        if value is None:
            value = float(0)

        amkey_values[amkey] = value

    return amkey_values


def make_submission():
    """
    Make a submission for the competition.

    The submission is saved to `submission.csv`.
    """
    if os.path.exists(SUBMISSION_LOG_FILE):
        os.remove(SUBMISSION_LOG_FILE)
    logger.remove()
    logger.add(SUBMISSION_LOG_FILE, level="DEBUG")

    company_submission_dfs = []

    for company in COMPANIES:
        all_years = list(COMPANY_YEAR_PDF_MAPPING[company].keys())
        submission_year = max(all_years)
        logger.info(f"Making submission for {company} {submission_year}")
        company_amkey_values = retrieve_all_amkey_values(company, submission_year)

        if company in ["Oceana", "Uct"]:
            company_str = company + "1&2"
        else:
            company_str = company

        company_submission_df = pd.DataFrame(
            {
                "ID": [f"{amkey}_X_{company_str}" for amkey in company_amkey_values.keys()],
                "2022_Value": list(company_amkey_values.values()),
            }
        )

        company_submission_dfs.append(company_submission_df)

    submission_df = pd.concat(company_submission_dfs, ignore_index=True)

    submission_df.to_csv(Path(__file__).parent / "submission.csv", index=False)
    logger.info("Submission created")


if __name__ == "__main__":
    make_submission()
