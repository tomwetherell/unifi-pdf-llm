"""Module containing functions to preprocess parsed pdfs."""

import pandas as pd
from haystack import Document
from haystack.nodes import PreProcessor


def preprocess_documents(
        docs: list[Document],
        window_size: int=5,
        discard_text: bool=True
    ) -> list[Document]:
    """
    Preprocess the documents.

    The documents are assumed to be the output of `haystack.nodes.AzureConverter`.

    Parameters
    ----------
    docs : list[Document]
        The documents to preprocess.

    window_size : int
        The size of the sliding window used to split the tables.

    discard_text : bool
        If True, discard text passages and keep only tables.

    Returns
    -------
    docs : list[Document]
        The preprocessed documents.
    """
    preprocessed_docs = []

    # Preprossor used to split text documents
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=50,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000
    )

    for doc in docs:
        if doc.content_type == "table":
            doc.content = clean_table_column_names(doc.content)
            doc.content = clean_table_values(doc.content)
            sliced_table_docs = slice_table_document(doc, window_size)
            preprocessed_docs.extend(sliced_table_docs)
        else:
            if discard_text:
                continue
            split_text_docs = processor.process([doc])
            preprocessed_docs.extend(split_text_docs)

    convert_tables_to_markdown(preprocessed_docs)

    return preprocessed_docs


def clean_table_column_names(df: pd.DataFrame, replace: str=' - ') -> pd.DataFrame:
    """
    Return a DataFrame with newlines removed from column headers.

    Parameters
    ----------
    df : pd.Dataframe
        The DataFrame to clean.

    replace: str
        The string to replace newlines with.

    Returns
    -------
    df : pd.Dataframe
        The dataframe with newlines removed from column headers.
    """
    df.columns = df.columns.str.replace('\n', replace)
    return df


def clean_table_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with commas and spaces replaced or removed from values.

    Commas are replaced with a decimal point, and spaces are removed.

    Parameters
    ----------
    df : pd.Dataframe
        The DataFrame to clean.

    Returns
    -------
    df : pd.Dataframe
        The dataframe with commas and spaces replaced or removed from values.
    """
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x).replace(',', '.').replace(' ', '') if _is_number(str(x)) else x
        )
    return df


def _is_number(string: str) -> bool:
    """
    Return True if the string is a number, False otherwise.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_number : bool
        True if the string is a number, False otherwise.
    """
    is_number = string.replace('.','').replace(',', '').replace(' ', '').isdigit()
    return is_number


def slice_table_document(doc: Document, window_size: int=5) -> list[Document]:
    """
    Return a list of documents, each containing a table with `window_size` rows.

    A sliding window approach is used to split the table into smaller tables. The
    returned documents have the same metadata as the original document, except for
    the content and id.

    Parameters
    ----------
    doc : Document
        Document with content_type "table".

    window_size : int
        The size of the sliding window.

    Returns
    -------
    docs : list[Document]
        A list of documents, each one containing a table with `window_size` rows.

    Raises
    ------
    ValueError
        If the document does not contain a table.
    """
    if doc.content_type != "table":
        raise ValueError("The document does not contain a table.")

    tables = _sliding_window(doc.content, window_size)
    docs = []
    for table in tables:
        new_doc = Document(content=table)
        for attr, value in doc.__dict__.items():
            if attr not in ["content", "id"]:
                setattr(new_doc, attr, value)
        docs.append(new_doc)

    return docs


def _sliding_window(df: pd.DataFrame, window_size: int) -> list[pd.DataFrame]:
    """
    Return a list of DataFrames, each containing a window of the original DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to split.

    window_size : int
        The size of the sliding window.

    Returns
    -------
    tables : list[pandas.DataFrame]
        A list of DataFrames, each containing a window of the original DataFrame.
    """
    tables = [df.iloc[i:i+window_size] for i in range(len(df) - window_size + 1)]

    return tables


def convert_tables_to_markdown(docs: list[Document]) -> None:
    """
    Convert tables to markdown format in place.

    Parameters
    ----------
    docs : List[Document]
        List of Documents, some of which may have `content_type` 'table'.
    """
    for doc in docs:
        if doc.content_type == "table":
            _convert_table_to_markdown(doc)


def _convert_table_to_markdown(doc: Document) -> None:
    """
    Convert table to markdown format in place.

    Parameters
    ----------
    doc : Document
        Document with `content_type` table.

    Raises
    ------
    ValueError
        If `doc.content_type` is not "table".
    """
    if doc.content_type != "table":
        raise ValueError(f"Document content_type must be 'table', not '{doc.content_type}'")

    table = doc.content
    markdown_table = table.to_markdown(index=False, tablefmt="github", intfmt='')

    doc.content = markdown_table
    doc.content_type = "text"
