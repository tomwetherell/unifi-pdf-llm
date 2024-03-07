"""
Script to validate the performance of the end-to-end RAG system.

TODO: Argparse, allow user to specify the company and year to validate. By default,
run the full validation (as is currently the case).
"""

import os

import pandas as pd
from loguru import logger

from esg_retriever.rag.load import load_documents
from esg_retriever.rag.preprocess import preprocess_documents
from esg_retriever.rag.rag import ModularRAG


VALIDATION_LOGS_PATH = "/home/tomw/unifi-pdf-llm/esg_retriever/validation_results"
"""Path to the directory where the validation logs are saved."""

TRAIN_CSV_PATH = "/home/tomw/unifi-pdf-llm/esg_retriever/data/Train.csv"
"""Path to the Train.csv file."""

VALIDATION_COMPANY_YEAR_PAIRS = [
    ("Absa", 2021),
    ("Distell", 2021),
    ("Picknpay", 2021),
    ("Sasol", 2021),
    ("Ssw", 2021),
    ("Tongaat", 2021),
    ("Uct", 2021),
]
"""
List of company and year pairs to validate the performance of the end-to-end
RAG system on.
"""


def run_validation():
    """Run the validation tests."""
    validation_results = pd.DataFrame(columns=["Company", "Year", "Validation Type", "Accuracy w/ Validation", "Accuracy w/o Validation", "Num"])

    for company, year in VALIDATION_COMPANY_YEAR_PAIRS:
        log_filename = f"{VALIDATION_LOGS_PATH}/{company}_{year}.log"

        if os.path.exists(log_filename):
            os.remove(log_filename)

        logger.remove()
        logger.add(log_filename, level="DEBUG")

        for validation_type in ["retrieval", "nan"]:
            logger.info(f"\nValidation type:  {validation_type}\n")

            accuracy_w_validation, accuracy_wo_validation, results_df = validate_retrieval(
                company, year, type=validation_type, num=50, window_size=2, discard_text=True
            )

            number_of_preds = len(results_df)

            validation_results = pd.concat(
                [
                    validation_results,
                    pd.DataFrame(
                        {
                            "Company": [company],
                            "Year": [year],
                            "Validation Type": [validation_type],
                            "Accuracy w/ Validation": [accuracy_w_validation],
                            "Accuracy w/o Validation": [accuracy_wo_validation],
                            "Num": [number_of_preds],
                        }
                    ),
                ],
                ignore_index=True,
            )

            results_df_markdown = results_df.to_markdown(index=False, tablefmt="github", intfmt="")
            logger.info(f"Results:\n\n{results_df_markdown}\n")

    validation_log_filename = f"{VALIDATION_LOGS_PATH}/validation_results.log"

    if os.path.exists(validation_log_filename):
        os.remove(validation_log_filename)

    logger.remove()
    logger.add(validation_log_filename, level="INFO")

    # Reorder validation results, so that rows corresponding to retrieval_type = "retrieval" come first
    validation_results = validation_results.sort_values(by="Validation Type", ascending=False)
    validation_results_markdown = validation_results.to_markdown(index=False, tablefmt="github", intfmt="")

    logger.info(f"Validation results:\n{validation_results_markdown}")

    retrieval_accuracy_w_validation = (validation_results[validation_results["Validation Type"] == "retrieval"]["Accuracy w/ Validation"] * validation_results[validation_results["Validation Type"] == "retrieval"]["Num"]).sum() / validation_results[validation_results["Validation Type"] == "retrieval"]["Num"].sum()
    retrieval_accuracy_wo_validation = (validation_results[validation_results["Validation Type"] == "retrieval"]["Accuracy w/o Validation"] * validation_results[validation_results["Validation Type"] == "retrieval"]["Num"]).sum() / validation_results[validation_results["Validation Type"] == "retrieval"]["Num"].sum()
    nan_accuracy_w_validation = (validation_results[validation_results["Validation Type"] == "nan"]["Accuracy w/ Validation"] * validation_results[validation_results["Validation Type"] == "nan"]["Num"]).sum() / validation_results[validation_results["Validation Type"] == "nan"]["Num"].sum()
    nan_accuracy_wo_validation = (validation_results[validation_results["Validation Type"] == "nan"]["Accuracy w/o Validation"] * validation_results[validation_results["Validation Type"] == "nan"]["Num"]).sum() / validation_results[validation_results["Validation Type"] == "nan"]["Num"].sum()

    logger.info(f"Average accuracy w/ validation (retrieval): {retrieval_accuracy_w_validation}")
    logger.info(f"Average accuracy w/o validation (retrieval): {retrieval_accuracy_wo_validation}")
    logger.info(f"Average accuracy w/ validation (nan): {nan_accuracy_w_validation}")
    logger.info(f"Average accuracy w/o validation (nan): {nan_accuracy_wo_validation}")

def validate_retrieval(
    company: str,
    year: int,
    type: str="retrieval",
    num: int=50,
    window_size: int=1,
    discard_text: bool=True
) -> tuple[float, float]:
    """
    Returns the accuracy of the RAG system with and without the validation step.

    TODO: The accuracy with validation also includes unit conversion. Which means
    with type 'retrieval', the accuracy with validation can actually be higher than
    the accuracy without validation (shouldn't happen).

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
    accuracy_w_validation : float
        The accuracy of the RAG system with the validation step.

    accuracy_wo_validation : float
        The accuracy of the RAG system without the validation step.

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

        value, unvalidated_value = query_pipeline.query(amkey, year)
        results_df.at[idx, f"{year}_Generated"] = value
        results_df.at[idx, f"{year}_Gen_Unvalidated"] = unvalidated_value

    results_df[f"{year}_Value"] = results_df[f"{year}_Value"].astype(float)
    results_df[f"{year}_Generated"] = results_df[f"{year}_Generated"].astype(float)
    results_df["Correct"] = results_df.apply(
        lambda row: (row[f"{year}_Generated"] == row[f"{year}_Value"]) or
        (pd.isna(row[f"{year}_Generated"]) and pd.isna(row[f"{year}_Value"])) or
        (row[f"{year}_Generated"] == -1 and pd.isna(row[f"{year}_Value"])),
        axis=1
    )

    # Reordering the columns
    results_df = results_df[["ID", "Metric", f"{year}_Value", f"{year}_Gen_Unvalidated", f"{year}_Generated", "Correct"]]

    accuracy_w_validation = results_df["Correct"].sum() / len(results_df)

    logger.info(f"Accuracy w/ validation: {accuracy_w_validation}")

    accurcy_wo_validation = results_df.apply(
        lambda row: (row[f"{year}_Gen_Unvalidated"] == row[f"{year}_Value"]) or
        (pd.isna(row[f"{year}_Gen_Unvalidated"]) and pd.isna(row[f"{year}_Value"])),
        axis=1
    ).sum() / len(results_df)

    logger.info(f"Accuracy w/o validation: {accurcy_wo_validation}")

    return accuracy_w_validation, accurcy_wo_validation, results_df


if __name__ == "__main__":
    run_validation()
