"""Module containing functions to preprocess parsed pdfs."""

import os
import json
import re

import pandas as pd
from openai import OpenAI
from haystack import Document
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from dotenv import load_dotenv
from loguru import logger

from esg_retriever.rag.prompts import (
    FILTER_CONTEXT_PROMPT_TEMPLATE,
    RETRIEVE_VALUE_PROMPT_TEMPLATE,
    VALIDATE_RESPONSE_PROMPT_TEMPLATE,
    UNIT_CONVERSION_PROMPT_TEMPLATE,
)


AMKEY_TO_METRIC_PATH = (
    "/home/tomw/unifi-pdf-llm/esg_retriever/data/AMKEY_GoldenStandard.csv"
)
"""Path to csv file mapping AMKEY to metric description."""

AMKEY_TO_SYNONYM_PATH = (
    "/home/tomw/unifi-pdf-llm/esg_retriever/data/ActivityMetricsSynonyms.csv"
)
"""Path to csv file mapping AMKEY to company metric description."""

AMKEY_TO_UNIT_PATH = (
    "/home/tomw/unifi-pdf-llm/esg_retriever/data/AMKEY_unit_conversion.csv"
)
"""Path to csv file mapping AMKEY to required unit."""
# TODO: Centralise paths


load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class ModularRAG:
    """Class implementing a modular retrieval augmented generation pipeline."""

    def __init__(
        self,
        docs: list[Document],
        company: str,
        top_k: int = 3,
        amkey_to_metric_path: str = AMKEY_TO_METRIC_PATH,
        amkey_to_synonym_path: str = AMKEY_TO_SYNONYM_PATH,
        amkey_to_unit_path: str = AMKEY_TO_UNIT_PATH,
    ):
        """
        Initalise the components of the query pipeline.

        Parameters
        ----------
        docs : list[Document]
            The documents to provide context for the queries.

        company : str
            The company the documents are for.

        top_k : int
            The number of documents to retrieve for each query.

        amkey_to_metric_path : str
            Path to a csv file mapping AMKEY to metric.

        amkey_to_synonym_path : str
            Path to a csv file mapping AMKEY and company to metric synonym.

        amkey_to_unit_path : str
            Path to a csv file mapping AMKEY to desired unit.
        """
        self.docs = docs
        self.company = company

        # Initalised in the methods below
        self.document_store = None
        self.retriever = None
        self.unit_conversion_llm = None
        self.amkey_to_metric_df = None
        self.amkey_to_synonym_df = None
        self.amkey_to_unit_df = None

        self._initialise_document_store()
        self._initialise_retriever(top_k)
        self._initialise_mappings(
            amkey_to_metric_path, amkey_to_synonym_path, amkey_to_unit_path
        )

    def _initialise_document_store(self):
        # TODO: Try using other document stores (e.g. FAISS).
        logger.info("Initialising document store")
        # Embedding dimension of the document store must match the dimension of the
        # retriever's embedding model.
        self.document_store = InMemoryDocumentStore(embedding_dim=768, similarity="cosine")
        self.document_store.delete_documents()
        self.document_store.write_documents(self.docs)

    def _initialise_retriever(self, top_k: int):
        """
        Initialise the (dense) retriever.

        Parameters
        ----------
        top_k : int
            The number of documents to retrieve for each query.
        """
        # TODO: Look into other (non-OpenAI) embedding models that can be used with
        # Haystack v1.
        logger.info("Initialising retriever (using all-mpnet-base-v2 model)")
        self.retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            document_store=self.document_store,
            top_k=top_k,
        )
        self.document_store.update_embeddings(retriever=self.retriever)

    def _initialise_mappings(
        self,
        amkey_to_metric_path: str,
        amkey_to_synonym_path: str,
        amkey_to_unit_path: str,
    ):
        """
        Initialise mapping dataframes.

        Parameters
        ----------
        amkey_to_metric_path : str
            Path to a csv file mapping AMKEY to metric.

        amkey_to_synonym_path : str
            Path to a csv file mapping AMKEY and company to metric synonym.

        amkey_to_unit_path : str
            Path to a csv file mapping AMKEY to desired unit.
        """
        logger.info("Initialising mappings")
        self.amkey_to_metric_df = pd.read_csv(amkey_to_metric_path)
        self.amkey_to_synonym_df = pd.read_csv(amkey_to_synonym_path)
        self.amkey_to_unit_df = pd.read_csv(amkey_to_unit_path)

    def query(self, amkey: int, year: int) -> tuple[float | None, float | None]:
        """
        Return the value associated with an AMKEY for a given year.

        Uses retrieval augmented generation to retrieve the value.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric to retrieve.

        year : int
            The year to retrieve the metric for.

        Returns
        -------
        value : float | None
            The value associated with the AMKEY for the given year. If the value
            cannot be retrieved, None is returned.

        unvalidated_value : float | None
            The value associated with the AMKEY for the given year, before validation.
            If the value cannot be retrieved, None is returned.
        """
        # Retrieving relevant context documents
        logger.debug(f"Retrieving AMKEY: {amkey}")
        metric = self.retrieve_metric_description(amkey)
        logger.debug(f"Retrieving metric: {metric}")
        context_documents = self.retriever.retrieve(metric)
        additional = self._retrieve_additional_appended_instructions(amkey)
        question = f"What was the {metric} in the year {year}? {additional}"
        relevant_context_documents = self.filter_context_documents(context_documents, question)

        if not relevant_context_documents:
            logger.info("No relevant context documents found")
            return None, None

        # Generation
        answer = self.retrieve_value(question, relevant_context_documents)

        try:
            value, unit = self.parse_answer(answer)
        except Exception as err:
            logger.error(f"Exception raised when parsing answer: {err}")
            value, unit = None, None

        logger.debug(f"Parsed answer: {value}, {unit}")

        # Validation
        # TODO: Consider validating the non-cleaned answer (for example, Level1 instead of 1.0)
        # The validator is returning 'no' for quite a few valid answers. Often, it wrongly states
        # that the value is provided for the other years, but not for the year in question.
        unvalidated_value = value
        if value is not None:
            logger.debug(f"Validating answer: {value}")
            valid_response = self.validate_response(question, value, relevant_context_documents)
            logger.debug(f"Valid response: {valid_response}")
            if valid_response == "no":
                value, unit = None, None

        # Unit conversion
        required_unit = self.retrieve_unit(amkey)
        logger.debug(f"Required unit: {required_unit}")

        if value is not None and required_unit is not None:
            if unit != required_unit:
                value = self.convert_unit(value, unit, required_unit)

        return value, unvalidated_value

    def retrieve_value(self, question: str, docs: list[Document]) -> str:
        """
        Return the value associated with a question from the context documents.

        Parameters
        ----------
        question : str
            The question to retrieve the value for.

        docs : list[Document]
            The context documents to retrieve the value from.

        Returns
        -------
        answer : str
            The answer to the question.
        """
        context = "\n\n".join([doc.content for doc in docs])

        prompt = RETRIEVE_VALUE_PROMPT_TEMPLATE.format(
            context=context, question=question
        )

        logger.debug(f"Retrieval prompt:\n{prompt}")

        # TODO: Test using `top_p` parameter. Not recommended to set both `temperature` and `top_p`.
        answer = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at reading markdown tables. Provide the answer in json format with the keys 'Answer' and 'Unit'.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.01,
                seed=0,
            )
            .choices[0]
            .message.content
        )

        logger.debug(f"Retrieval response: {answer}")

        return answer


    def filter_context_documents(
        self, docs: list[Document], question: str
    ) -> list[Document]:
        """
        Return the documents that are relevant to the question.

        Parameters
        ----------
        docs : list[Document]
            The context documents to filter.

        question : str
            The question to filter the context documents by.

        Returns
        -------
        relevant_docs : list[Document]
            The context documents that are relevant to the question.
        """
        relevant_docs = []

        for doc in docs:
            prompt = FILTER_CONTEXT_PROMPT_TEMPLATE.format(
                question=question, context=doc.content
            )

            logger.debug(f"Filtering context prompt:\n{prompt}")

            response = (
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at reading markdown tables",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.01,
                    seed=0,
                )
                .choices[0]
                .message.content
            )
            logger.debug(f"Filtering context response: {response}")

            conclusion = self._extract_conclusion(response)
            logger.debug(f"Filtering context conclusion: {conclusion}")

            if conclusion == "yes":
                relevant_docs.append(doc)

        return relevant_docs


    def validate_response(
        self, question: str, answer: float, docs: list[Document]
    ) -> str:
        """
        Returns 'yes' or 'no' to validate the response from the generation LLM.

        Parameters
        ----------
        question : str
            The question provided to the generation LLM.

        answer : float
            The answer provided by the generation LLM.

        docs : list[Document]
            The context documents provided to the generation LLM.

        Returns
        -------
        conclusion : str
            Whether the answer is valid or not. Either 'yes' or 'no'.
        """
        context = "\n\n".join([doc.content for doc in docs])

        prompt = VALIDATE_RESPONSE_PROMPT_TEMPLATE.format(
            context=context, answer=answer, question=question
        )

        logger.debug(f"Validation prompt:\n{prompt}")

        response = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at reading markdown tables",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01,
                seed=0,
            )
            .choices[0]
            .message.content
        )

        conclusion = self._extract_conclusion(response)

        logger.debug(f"Validation response: {response}")

        return conclusion

    def _extract_conclusion(self, response: str) -> str | None:
        """
        Extract 'yes' or 'no' conclusion from the response.

        Parameters
        ----------
        response : str
            The response from the model. This is expected to end with a 'yes' or 'no'
            conclusion.

        Returns
        -------
        conclusion : str | None
            The conclusion from the answer - either 'yes' or 'no'. If the conclusion
            is not found, None is returned.
        """
        conclusion = None

        # First, check if the answer ends with a 'yes' or 'no'
        conclusion = response.split()[-1].lower()

        if conclusion not in ["yes", "no"]:
            # Use regex to extract the conclusion
            matches = re.findall(r"\b(yes|no)\b", response, re.IGNORECASE)
            if matches:
                conclusion = matches[-1].lower()

        return conclusion

    def convert_unit(self, value: float, unit: str, target_unit: str) -> float:
        """
        Returns the value converted to the target unit.

        TODO: Consider whether this could be replaced with a hard-coded approach.

        Parameters
        ----------
        value : float
            The value to convert.

        unit : str
            The unit of the value.

        target_unit : str
            The unit to convert the value to.

        Returns
        -------
        converted_value : float
            The value converted to the target unit.
        """
        prompt = UNIT_CONVERSION_PROMPT_TEMPLATE.format(
            value=value, unit=unit, target_unit=target_unit
        )

        logger.debug(f"Unit conversion prompt:\n{prompt}")

        response = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert unit converter.  Provide the answer in json format with the key 'Answer'.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0.01,
                seed=0,
            )
            .choices[0]
            .message.content
        )

        response = json.loads(response)

        converted_value = float(response["Answer"])

        logger.debug(f"Unit conversion response: {response}")

        return converted_value

    def _retrieve_additional_appended_instructions(self, amkey: int) -> str:
        """
        Return additional instructions to append to the query.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric to retrieve.

        Returns
        -------
        additional : str
            Additional instructions to append to the query.
        """
        if amkey in [47, 48, 49]:
            additional = "Do not include the word 'Level' in the answer."
        else:
            additional = ""

        return additional

    def parse_answer(self, answer: str) -> tuple[float | None, str | None]:
        """
        Parse the answer returned by the generation LLM.

        Parameters
        ----------
        answer : str
            The answer returned by the generation LLM. This is expected to be in
            the format of a JSON object with the keys 'Answer' and 'Unit'.

        Returns
        -------
        value : float | None
            The value from the answer.

        unit : str | None
            The unit from the answer.
        """
        answer = json.loads(answer)

        value = answer["Answer"]
        unit = answer["Unit"]

        if value is None:
            pass
        elif isinstance(value, str) and value.lower() in ["null", "n/a"]:
            value = None
        elif value == "nil":
            value = 0
        elif isinstance(value, str):
            value = value.replace(" ", "").replace(",", "")
            value = "".join(filter(lambda x: x.isdigit() or x == ".", value))
            value = float(value)
        else:
            value = float(value)

        if unit == "null":
            unit = None

        return value, unit

    def retrieve_metric_description(self, amkey: int) -> str:
        """
        Return the description of a metric.

        If a company-specific description is available, it is returned. Otherwise, the
        generic description is returned.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric.

        Returns
        -------
        metric : str
            The description of the metric.
        """
        metric = self.retrieve_company_metric_description(amkey)
        if metric is None:
            metric = self.retrieve_generic_metric_description(amkey)

        return metric

    def retrieve_company_metric_description(self, amkey: int) -> str | None:
        """
        Return the company-specific description of a metric, if available.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric.

        Returns
        -------
        metric : str | None
            The company-specific description of the metric, if available.
            Otherwise, None.
        """
        metric = self.amkey_to_synonym_df[
            (self.amkey_to_synonym_df["AMKEY"] == amkey)
            & (self.amkey_to_synonym_df["Group"] == self.company)
        ]["ClientMetric"]

        if metric.empty:
            metric = None
        else:
            metric = metric.item()

        return metric

    def retrieve_generic_metric_description(self, amkey: int) -> str:
        """
        Return the generic description of a metric.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric.

        Returns
        -------
        metric : str
            The description of the metric.

        Raises
        ------
        ValueError
            If the AMKEY is invalid.
        """
        try:
            metric = self.amkey_to_metric_df[self.amkey_to_metric_df["AMKEY"] == amkey][
                "ActivityMetric"
            ].item()
        except Exception:
            raise ValueError(f"Invalid AMKEY {amkey}")

        return metric

    def retrieve_unit(self, amkey: int) -> str | None:
        """
        Return the required unit for a metric.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric.

        Returns
        -------
        unit : str | None
            The required unit for the metric, if specified. Otherwise, None.
        """
        try:
            unit = self.amkey_to_unit_df[self.amkey_to_unit_df["AMKEY"] == amkey][
                "Unit"
            ].item()
        except KeyError:
            unit = None

        if pd.isna(unit):
            unit = None

        return unit
