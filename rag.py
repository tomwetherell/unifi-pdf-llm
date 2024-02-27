"""Module containing functions to preprocess parsed pdfs."""

import os
import json
import re

import pandas as pd
from openai import OpenAI
from haystack import Document
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.document_stores import InMemoryDocumentStore

from loguru import logger

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

AMKEY_TO_METRIC_PATH = "/home/tomw/unifi-pdf-llm/data/AMKEY_GoldenStandard.csv"
"""Path to csv file mapping AMKEY to metric description."""

AMKEY_TO_SYNONYM_PATH = "/home/tomw/unifi-pdf-llm/data/ActivityMetricsSynonyms.csv"
"""Path to csv file mapping AMKEY to company metric description."""

AMKEY_TO_UNIT_PATH = "/home/tomw/unifi-pdf-llm/data/AMKEY_unit_conversion.csv"
"""Path to csv file mapping AMKEY to required unit."""


client = OpenAI()

RETRIEVE_VALUE_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
The answer must be a value from the context.
The context may be text or a markdown table.
Just retrieve the answer from the context. Please don't do any unit conversion.
If you don't know the answer, please return 'null' for the answer and unit.
Do not return any words other than 'Answer' and 'Unit' in the answer.
Please return the answer in the format of a python dictionary / JSON object:
{{"Answer": <number or null>, "Unit": <unit or null>}}
Please always use double quotes for the keys and values.
If the requested value is not present in the context, please return 'null' for the answer and unit.

Context:

{context}

Question: {question} {append}

Answer:
"""

VALIDATE_RESPONSE_PROMPT_TEMPLATE ="""
Consider the following markdown tables:

{context}

Are you sure that '{answer}' is the correct answer to the question: "{question}"?

It is possible that the answer is not explicitly stated in the context.

Think step by step. Please conclude your answer with a 'yes' or 'no'.

Answer:
"""


class ModularRAG:
    """Class implementing a modular retrieval augmented generation pipeline."""

    def __init__(
            self,
            docs: list[Document],
            company: str,
            top_k: int=3,
            amkey_to_metric_path: str=AMKEY_TO_METRIC_PATH,
            amkey_to_synonym_path: str=AMKEY_TO_SYNONYM_PATH,
            amkey_to_unit_path : str=AMKEY_TO_UNIT_PATH,
        ):
        """
        Initalise the components of the query pipeline.

        Parameters
        ----------
        docs : list[Document]
            The documents to provide context for the queries.

        company : str
            The company the documents are for.

        amkey_to_metric_path : str
            Path to a csv file mapping AMKEY to metric.

        amkey_to_synonym_path : str
            Path to a csv file mapping AMKEY and company to metric synonym.

        amkey_to_unit_path : str
            Path to a csv file mapping AMKEY to desired unit.
        """
        self.docs = docs
        self.company = company
        self.top_k = top_k
        self.amkey_to_metric_path = amkey_to_metric_path
        self.amkey_to_synonym_path = amkey_to_synonym_path
        self.amkey_to_unit_path = amkey_to_unit_path

        self.document_store = None
        self.retriever = None
        self.generation_llm = None
        self.unit_conversion_llm = None
        self.json_conversion_llm = None
        self.confirm_relevant_context_llm = None
        self.amkey_to_metric = None
        self.amkey_to_synonym = None
        self.amkey_to_unit = None

        self.initialise_document_store()
        self.initialise_retriever()
        self.initialise_generation_llm()
        self.initialise_unit_conversion_llm()
        self.initialise_json_conversion_llm()
        self.initialise_relevant_context_llm()
        self.initialise_mappings()

    def initialise_document_store(self):
        # TODO: Try using other document stores (e.g. FAISS).
        logger.info("Initialising document store")
        self.document_store = InMemoryDocumentStore(embedding_dim=384)
        self.document_store.delete_documents()
        self.document_store.write_documents(self.docs)

    def initialise_retriever(self):
        # TODO: I'm not sure which OpenAI embedding models are available. Is it possible
        # to use their newest embedding models in Haystack v1?
        # TODO: Look into other (non-OpenAI) embedding models that can be used with
        # Haystack v1.
        logger.info("Initialising retriever")
        self.retriever = EmbeddingRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            document_store=self.document_store,
            top_k=self.top_k
        )
        self.document_store.update_embeddings(retriever=self.retriever)

    def initialise_generation_llm(self):
        # TODO: Temperature = 0 isn't giving deterministic results. Is this a bug?
        # TODO: Currently 'gpt-3.5-turbo-1106' as it has a larger context window.
        # There is a newer gpt-3.5 model (gpt-3.5-turbo-0125) which also has a large
        # context window. I should use that if it's available with Haystack v1.
        logger.info("Initialising generation LLM")
        self.generation_llm = PromptNode(
            model_name_or_path="gpt-3.5-turbo-1106",
            api_key=OPENAI_API_KEY,
            model_kwargs={"temperature": 0}
        )

    def initialise_unit_conversion_llm(self):
        logger.info("Initialising unit conversion LLM")
        self.unit_conversion_llm = PromptNode(
            model_name_or_path="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            model_kwargs={"temperature": 0}
        )

    def initialise_json_conversion_llm(self):
        logger.info("Initialising json conversion LLM")
        self.json_conversion_llm = PromptNode(
            model_name_or_path="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            model_kwargs={"temperature": 0}
        )

    def initialise_relevant_context_llm(self):
        logger.info("Initialising relevant context LLM")
        self.confirm_relevant_context_llm = PromptNode(
            model_name_or_path="gpt-3.5-turbo",
            api_key=OPENAI_API_KEY,
            model_kwargs={"temperature": 0.4}
        )

    def initialise_mappings(self):
        logger.info("Initialising mappings")
        self.amkey_to_metric = pd.read_csv(self.amkey_to_metric_path)
        self.amkey_to_synonym = pd.read_csv(self.amkey_to_synonym_path)
        self.amkey_to_unit = pd.read_csv(self.amkey_to_unit_path)

    def query(self, amkey: int, year: int):
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
        value : int
            The value associated with the AMKEY for the given year.
        """
        logger.debug(f"Retrieving AMKEY: {amkey}")
        metric = self.retrieve_metric_description(amkey)
        logger.debug(f"Retrieving metric: {metric}")
        context_documents = self.retriever.retrieve(metric)
        append = self._retrieve_additional_appended_instructions(amkey)
        question = f"What was the {metric} in the year {year}?"
        answer = self.retrieve_value(question, append, context_documents)

        try:
            value, unit = self.parse_answer(answer)
        except json.JSONDecodeError:
            logger.error(f"Error parsing answer: {answer}")
            json_conversion_prompt = self.create_json_generation_prompt(answer)
            answer = self.json_conversion_llm(json_conversion_prompt)[0]
            logger.debug(f"JSON converted answer: {answer}")
            value, unit = self.parse_answer(answer)
        except Exception as err:
            logger.error(f"Non-JSONDecodeError exception when parsing answer: {err}")
            value, unit = None, None

        logger.debug(f"Parsed answer: {value}, {unit}")

        # Validation
        # TODO: Consider validating the non-cleaned answer (for example, Level1 instead of 1.0)
        # The validator is returning 'no' for quite a few valid answers. Often, it wrongly states
        # that the value is provided for the other years, but not for the year in question.
        unvalidated_value = value
        if isinstance(value, float):
            logger.debug(f"Validating answer: {value}")
            valid_response = self.validate_response(question, value, context_documents)
            logger.debug(f"Valid response: {valid_response}")
            if valid_response == "no":
                value, unit = None, None

        required_unit = self.retrieve_unit(amkey)
        logger.debug(f"Required unit: {required_unit}")

        if required_unit is not None and value is not None:
            if unit != required_unit:
                unit_conversion_prompt = self.create_unit_conversion_prompt(value, unit, required_unit)
                value = self.unit_conversion_llm(unit_conversion_prompt)[0]
                logger.debug(f"Unit converted value: {value}")

        return value, unvalidated_value

    def retrieve_value(self, question: str, append: str, docs: list[Document]) -> str:
        """
        Return the value associated with a question from the context documents.

        Parameters
        ----------
        question : str
            The question to retrieve the value for.

        append : str
            Additional instructions to append to the query.

        docs : list[Document]
            The context documents to retrieve the value from.

        Returns
        -------
        answer : str
            The answer to the question.
        """
        context = "\n\n".join([doc.content for doc in docs])

        prompt = RETRIEVE_VALUE_PROMPT_TEMPLATE.format(
            context=context, question=question, append=append
        )

        logger.debug(f"Retrieval prompt:\n{prompt}")

        answer = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at reading markdown tables. Provide the answer in json format with the keys 'Answer' and 'Unit'."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=0.01,
            seed=0,
        ).choices[0].message.content

        logger.debug(f"Retrieval response: {answer}")

        return answer

    def validate_response(self, question: str, answer: float, docs: list[Document]) -> str:
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at reading markdown tables"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.01,
            seed=0,
        ).choices[0].message.content

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

        if conclusion not in ['yes', 'no']:
            # Use regex to extract the conclusion
            match = re.search(r'\b(yes|no)\b', response, re.IGNORECASE)
            if match: conclusion = match.group(0).lower()

        return conclusion

    def _retrieve_additional_appended_instructions(self, amkey: int) -> str:
        """
        Return additional instructions to append to the query.

        Parameters
        ----------
        amkey : int
            The AMKEY of the metric to retrieve.

        Returns
        -------
        append : str
            Additional instructions to append to the query.
        """
        if amkey in [47, 48, 49]:
            append = "Do not include the word 'Level' in the answer."
        else:
            append = ""

        return append

    def create_unit_conversion_prompt(self, value: int, unit: str, target_unit: str) -> str:
        prompt=f"""
        You are an expert unit converter. You are aware of how to convert
        between different units within the same system of measurement.
        For example, 1236 million = 1236 * 1 million = 1236 * 1000000 = 1236000000.
        For example, to convert from Rm to R, you would multiply by 1000000. This is because
        1 Rm = 1000000 R.
        Do not do any unit conversion if it is not necessary. That is, if the
        unit is already in the required unit, do not convert it.
        For example, 'What is 242353 Rands in rand? Answer: 242353' is the correct answer.
        Please return a single number as your answer. Do not elaborate or give
        any context.\n\n

        What is {value} {unit} in {target_unit}? \n\n Answer:"""

        return prompt

    def create_json_generation_prompt(self, answer: str) -> str:
        prompt=f"""The following answer was generated by a large language model: {answer}.
                   Please convert this answer to follow the python dictionary / JSON object format:
                   {{"Answer": <number or null>, "Unit": <unit or null>'}}
                   \n\n Answer:"""

        return prompt

    def parse_answer(self, answer: str) -> tuple[float | None, str | None]:
        """
        Parse the answer returned by the generation LLM.

        Parameters
        ----------
        answer : str
            The answer returned by the generation LLM. This is expected to be in
            the format of a python dictionary / JSON object:
            {"Answer": <number or null>, "Unit": <unit or null>}

        Returns
        -------
        value : float | None
            The value from the answer.

        unit : str | None
            The unit from the answer.
        """
        logger.debug(f"Parsing answer: {answer}")

        answer_dict = json.loads(answer)

        logger.info(f"answer_dict: {answer_dict}")

        value = answer_dict["Answer"]
        unit = answer_dict["Unit"]

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

        if unit == "null": unit = None

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
        metric = self.amkey_to_synonym[
            (self.amkey_to_synonym["AMKEY"] == amkey)
            & (self.amkey_to_synonym["Group"] == self.company)
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
            metric = self.amkey_to_metric[
                self.amkey_to_metric["AMKEY"] == amkey
            ]["ActivityMetric"].item()
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
            unit = self.amkey_to_unit[
                self.amkey_to_unit["AMKEY"] == amkey
            ]["Unit"].item()
        except KeyError:
            unit = None

        if pd.isna(unit): unit = None

        return unit
