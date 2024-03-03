"""Module containing prompts for the LLMs used in the RAG system."""

RETRIEVE_VALUE_PROMPT_TEMPLATE = """
Use the following markdown tables to as context to answer the question at the end.
The answer must be a value retrieved directly from the context. Please don't do any unit conversion.

It is possible that the answer is not explicitly stated in the context.
If it is not possible to retrieve the answer from the context, please return 'null' for the answer and unit.

Context:

{context}

Question: {question}

Answer:
"""

VALIDATE_RESPONSE_PROMPT_TEMPLATE = """
Consider the following markdown tables:

{context}

Are you sure that '{answer}' is the correct answer to the question: "{question}"?

Please consider each table individually. Only pay attention to the row corresponding to the
year in questin. Be careful not to incorrectly state that the value is provided for the other years, but not for the year in question (unless this is the case).
It is possible that the answer is not explicitly stated in the context.

Think step by step. Please conclude your answer with a 'yes' or 'no'.

Answer:
"""
# TODO: The validator often returns 'no' if the value is in one of the tables, but not the others.
# Modify prompt so the LLM understands that it's fine if the value is only in one of the tables.

UNIT_CONVERSION_PROMPT_TEMPLATE = """
You are aware of how to convert between different units within the same system of measurement.
For example, 1236 million = 1236 * 1 million = 1236 * 1000000 = 1236000000.
For example, to convert from Rm to R, you would multiply by 1000000. This is because 1 Rm = 1000000 R.
Do not do any unit conversion if it is not necessary. That is, if the
unit is already in the required unit, do not convert it.
For example, 'What is 242353 Rands in rand? Answer: 242353' is the correct answer.
Please return a single number as your answer. Do not elaborate or give
any context.

What is {value} {unit} in {target_unit}?

Answer:
"""
