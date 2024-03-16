"""Script to make submission for competition."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from esg_retriever.query import create_amkey_df
from esg_retriever.config import COMPANY_YEAR_PDF_MAPPING, COMPANIES


SUBMISSION_LOG_FILE = Path(__file__).resolve().parent / "submission.log"
"""File to save the logs from making the submission."""


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

        company_df = create_amkey_df(
            company, submission_year, save_path=Path(__file__).parent
        )

        if company in ["Oceana", "Uct"]:
            company_str = company + "1&2"
        else:
            company_str = company

        company_submission_df = pd.DataFrame(
            {
                "ID": company_df["AMKEY"].astype(str) + "_X_" + company_str,
                "2022_Value": company_df["Value"],
            }
        )

        company_submission_df["2022_Value"] = company_submission_df[
            "2022_Value"
        ].fillna(0.0)

        company_submission_dfs.append(company_submission_df)

    submission_df = pd.concat(company_submission_dfs, ignore_index=True)

    submission_df.to_csv(Path(__file__).parent / "submission.csv", index=False)
    logger.info("Submission created")


if __name__ == "__main__":
    make_submission()
