"""Script to validate the performance of the end-to-end RAG system."""

import os
import asyncio
import sys
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import pandas as pd
from loguru import logger

from esg_retriever.load import load_documents
from esg_retriever.preprocess import preprocess_documents
from esg_retriever.rag import ModularRAG
from esg_retriever.config import MISC_DATA_DIR


VALIDATION_LOGS_PATH = Path(__file__).resolve().parent / "results"
"""Path to the directory where the validation logs are saved."""

TRAIN_CSV_PATH = Path(MISC_DATA_DIR) / "Train.csv"
"""Path to the Train.csv file."""

ALL_COMPANIES = [
    "Absa",
    "Distell",
    "Picknpay",
    "Sasol",
    "Ssw",
    "Tongaat",
    "Uct",
]
"""List of companies to validate."""
# TODO: This should be centralised.

VALIDATION_YEAR = 2021
"""The year to use for validation."""


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the performance of the end-to-end RAG system.")
    parser.add_argument(
        "--companies",
        type=str,
        nargs='+',
        default=ALL_COMPANIES,
        help="The companies to validate. If 'all', validate all companies.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="The number metrics to retrieve.",
    )
    args = parser.parse_args()
    return args


# TODO: Add a function to simplify the logging setup.
async def run_validation(companies: list[str], num: int = 50):
    """
    Validate the performance of the end-to-end RAG system.

    Parameters
    ----------
    companies : list[str]
        The companies to validate.

    num : int
        The number of metrics to retrieve.
    """
    validation_results = pd.DataFrame(
        columns=[
            "Company",
            "Year",
            "Validation Type",
            "Accuracy w/ Validation",
            "Accuracy w/o Validation",
            "Num",
        ]
    )

    for company in companies:
        # Log to company-specific log file
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.info(f"Validating company: {company}, year: {VALIDATION_YEAR}")

        company_debug_log_fn = f"{VALIDATION_LOGS_PATH}/{company}_{VALIDATION_YEAR}.log"

        if os.path.exists(company_debug_log_fn):
            os.remove(company_debug_log_fn)

        logger.remove()
        logger.add(company_debug_log_fn, level="DEBUG")

        for validation_type in ["retrieval", "nan"]:
            logger.info(f"Validation type:  {validation_type}\n")

            accuracy, results_df = (
                await validate_retrieval(
                    company,
                    VALIDATION_YEAR,
                    type=validation_type,
                    num=num,
                    window_size=2,
                    discard_text=True,
                )
            )

            number_of_preds = len(results_df)

            validation_results = pd.concat(
                [
                    validation_results,
                    pd.DataFrame(
                        {
                            "Company": [company],
                            "Year": [VALIDATION_YEAR],
                            "Validation Type": [validation_type],
                            "Accuracy": [accuracy],
                            "Num": [number_of_preds],
                        }
                    ),
                ],
                ignore_index=True,
            )

            results_df_markdown = results_df.to_markdown(
                index=False, tablefmt="github", intfmt=""
            )
            logger.info(f"Results:\n\n{results_df_markdown}\n")

    # If all companies have been validated, update the validation results log
    if companies == ALL_COMPANIES:
        validation_results_log_fn = f"{VALIDATION_LOGS_PATH}/validation_results.log"

        if os.path.exists(validation_results_log_fn):
            os.remove(validation_results_log_fn)

        logger.remove()
        logger.add(validation_results_log_fn, level="INFO")

        # Reorder validation results, so that rows corresponding to retrieval_type = "retrieval" come first
        validation_results = validation_results.sort_values(
            by="Validation Type", ascending=False
        )
        validation_results_markdown = validation_results.to_markdown(
            index=False, tablefmt="github", intfmt=""
        )

        logger.info(f"Validation results:\n{validation_results_markdown}")

        retrieval_accuracy = (
            validation_results[validation_results["Validation Type"] == "retrieval"][
                "Accuracy"
            ]
            * validation_results[validation_results["Validation Type"] == "retrieval"][
                "Num"
            ]
        ).sum() / validation_results[validation_results["Validation Type"] == "retrieval"][
            "Num"
        ].sum()

        nan_accuracy = (
            validation_results[validation_results["Validation Type"] == "nan"][
                "Accuracy"
            ]
            * validation_results[validation_results["Validation Type"] == "nan"]["Num"]
        ).sum() / validation_results[validation_results["Validation Type"] == "nan"][
            "Num"
        ].sum()

        logger.info(
            f"Average accuracy (retrieval): {retrieval_accuracy}"
        )
        logger.info(f"Average accuracy (nan): {nan_accuracy}")


async def validate_retrieval(
    company: str,
    year: int,
    type: str = "retrieval",
    num: int = 50,
    window_size: int = 2,
    discard_text: bool = True,
) -> tuple[float, float]:
    """
    Returns the accuracy of the RAG system.

    Parameters
    ----------
    company : str
        The company to validate.

    year : int
        The year to validate. Can be 2019, 2020, or 2021.

    type : str
        The type of validation test to run. Options are "retrieval" or "nan".
        The "retrieval" test checks the retrieval of values that are present in the
        documents. The "nan" test checks the retrieval of values that are not present
        in the documents (i.e. testing the ability to return 'None' when the value is
        not present).

    num : int
        The number of rows to validate.

    window_size : int
        The size of the sliding window to use when slicing tables.

    discard_text : bool
        If True, discard text passages when preprocessing the documents. Only tables
        are kept.

    Returns
    -------
    accuracy : float
        The accuracy of the RAG system.

    results_df : pd.DataFrame
        The results of the validation test for the given company and year. Contains
        the following columns: "ID", "Metric", f"{year}_Value", f"{year}_Generated",
        "Correct".

    Raises
    ------
    ValueError
        If the year is not 2019, 2020, or 2021. Or if the type is not "retrieval" or "nan".
    """
    if year not in [2019, 2020, 2021]:
        raise ValueError(f"Unable to validate year: {year}")

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # Restrict to the company
    train_df = train_df[train_df["ID"].str.contains(f"X_{company}")]
    train_df.reset_index(drop=True, inplace=True)

    # Drop the two columns that we are not interested in
    all_years = ["2021", "2020", "2019"]
    all_years.remove(str(year))
    for _year in all_years:
        train_df.drop(columns=[f"{_year}_Value"], inplace=True)

    if type == "retrieval":
        train_df = train_df.dropna(subset=[f"{year}_Value"], how="all")
    elif type == "nan":
        train_df = train_df[train_df[f"{year}_Value"].isna()]
    else:
        raise ValueError(f"Invalid validation type: {type}")

    # TODO: Should this be a random (with set seed) sample?
    train_df = train_df.head(n=num)

    # Load and preprocess the documents
    docs = load_documents(company, year)
    docs = preprocess_documents(
        docs, window_size=window_size, discard_text=discard_text
    )

    logger.debug(f"Number of documents: {len(docs)}")

    query_pipeline = ModularRAG(
        docs=docs,
        company=company,
    )

    results_df = train_df.copy(deep=True)

    # Loop over the rows in the dataframe and retrieve the value for each AMKEY
    for idx, row in train_df.iterrows():
        amkey = int(row["ID"].split("_")[0])

        metric = query_pipeline.retrieve_metric_description(amkey)
        results_df.at[idx, "Metric"] = metric

        value = await query_pipeline.query(amkey, year)
        results_df.at[idx, f"{year}_Generated"] = value

    await query_pipeline.close()

    results_df[f"{year}_Value"] = results_df[f"{year}_Value"].astype(float)
    results_df[f"{year}_Generated"] = results_df[f"{year}_Generated"].astype(float)
    results_df["Correct"] = results_df.apply(
        lambda row: (row[f"{year}_Generated"] == row[f"{year}_Value"])
        or (pd.isna(row[f"{year}_Generated"]) and pd.isna(row[f"{year}_Value"]))
        or (row[f"{year}_Generated"] == -1 and pd.isna(row[f"{year}_Value"])),
        axis=1,
    )

    # Reordering the columns
    results_df = results_df[
        [
            "ID",
            "Metric",
            f"{year}_Value",
            f"{year}_Generated",
            "Correct",
        ]
    ]

    accuracy = results_df["Correct"].sum() / len(results_df)

    logger.info(f"Accuracy: {accuracy}")

    return accuracy, results_df


if __name__ == "__main__":
    args = parse_args()
    num = args.num
    companies = args.companies
    asyncio.run(run_validation(companies, num))
