"""Module containing prompts for the LLMs used in the RAG system."""

# -------------------------- Filtering Context --------------------------

FILTER_CONTEXT_PROMPT_TEMPLATE = """
Can the answer to the specific question: "{question}" be directly found in the following markdown table:

{context}

It is possible that the answer is not explicitly stated in the context.
The answer, if it exists, should be specific to the question asked.
No calculations will be required to answer the question - the answer, if it exists, will be directly retrievable from the context.
If the answer is not directly retrievable from the context, your conclusion should be 'no'.
Think step by step, and explain your reasoning.
Please conclude your answer with a 'yes' or 'no'.

Answer:
"""

# Example for few-shot prompting
FILTER_CONTEXT_EXAMPLE_CONTEXT = """
| Indicator                         |   Trend |    2022 |    2021 |    2020 |    2019 |
|-----------------------------------|---------|---------|---------|---------|---------|
| Average hours training per person |       1 |   48.47 |   55.33 |   56.42 |   39.24 |
| Number of programmes accessed     |       1 | 7035    | 7294    | 5047    | 4719    |
"""
FILTER_CONTEXT_EXAMPLE_QUESTION = "What was the Average number of hours of health, safety, and emergency response training for contract employees in the year 2021?"
FILTER_CONTEXT_EXAMPLE_ANSWER = """
To find the average number of hours of health, safety, and emergency response training for contract employees in the year 2021, we need to look for a specific indicator related to this type of training in the provided markdown table. The table only includes indicators such as "Average hours training per person" and "Number of programmes accessed."

While the "Average hours training per person" indicator gives us the average hours of training per person, it does not specify whether this training includes health, safety, and emergency response training for contract employees. Without a specific breakdown of the types of training included in the average hours, we cannot determine the exact number of hours dedicated to health, safety, and emergency response training for contract employees in 2021.

Therefore, the answer to the question "What was the Average number of hours of health, safety, and emergency response training for contract employees in the year 2021?" is not directly retrievable from the context.

Conclusion: no
"""

# -------------------------- Retrieving Value --------------------------

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

# -------------------------- Validating Response --------------------------

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
# TODO: Consider whether validation is necessary now that we filter the context.

# -------------------------- Unit Conversion --------------------------

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
