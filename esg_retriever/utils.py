"""Utility functions."""

import pandas as pd

from esg_retriever.config import MISC_DATA_DIR


def retrieve_amkeys_list() -> list[int]:
    """
    Return list of all Activity Metric Keys (AMKEYs).
    """
    amkeys_df = pd.read_csv(MISC_DATA_DIR / "AMKEY_GoldenStandard.csv")

    amkeys_list = amkeys_df["AMKEY"].tolist()

    return amkeys_list
