"""Query AMKEY values for a company and year."""

import argparse
from pathlib import Path

import pandas as pd

from esg_retriever.load import load_and_preprocess_documents
from esg_retriever.rag import ModularRAG
from esg_retriever.utils import list_all_amkeys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve AMKEY values for a company and year, and save them to "
        " a .csv file.",
    )

    parser.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company name.",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the AMKEY values as a .csv file. The name of the file will "
        "be {company}_{year}_amkey_values.csv.",
    )

    args = parser.parse_args()

    return args


def retrieve_all_amkey_values(company: str, year: int) -> dict[int, float | None]:
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
        A dictionary with the AMKEY as the key and the retrieved value as the value.
        If the value is not found, it is set to None.
    """
    amkey_values = {}
    amkeys = list_all_amkeys()

    docs = load_and_preprocess_documents(
        company, year, window_size=2, discard_text=True
    )
    rag = ModularRAG(docs=docs, company=company)

    for amkey in amkeys:
        value = rag.query(amkey, year=year)
        amkey_values[amkey] = value

    return amkey_values


def create_amkey_df(
    company: str, year: int, save_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Return DataFrame with retrieved AMKEY values for a company and year.

    Parameters
    ----------
    company : str
        The company name.

    year : int
        The year to retrieve the AMKEY values for.

    save_path : str | Path | None, default=None
        If not None, save the DataFrame as a CSV file to the path. The name of the
        csv file saved to disk will be `{company}_{year}_amkey_values.csv`.

    Returns
    -------
    amkey_df : pd.DataFrame
        A DataFrame with columns 'Company', 'Year', 'AMKEY', and 'Value'. The
        DataFrame contains the retrieved AMKEY values for the company and year. If
        the value associated with an AMKEY is not found, it is set to NaN in the
        dataframe.
    """
    amkey_values = retrieve_all_amkey_values(company, year)
    amkey_df = pd.DataFrame(amkey_values.items(), columns=["AMKEY", "Value"])
    amkey_df["Company"] = company
    amkey_df["Year"] = year
    amkey_df = amkey_df[["Company", "Year", "AMKEY", "Value"]]

    if save_path is not None:
        save_path = Path(save_path)
        filename = f"{company}_{year}_amkey_values.csv"
        amkey_df.to_csv(save_path / filename, index=False)

    return amkey_df


if __name__ == "__main__":
    args = parse_args()
    company = args.company
    year = args.year
    save_path = args.save_path
    create_amkey_df(company, year, save_path=save_path)
