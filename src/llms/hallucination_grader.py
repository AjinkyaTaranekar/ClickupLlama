from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an expert fact-checker tasked with evaluating the factual accuracy of an answer based on provided documents. Your goal is to determine if the answer contains any information not supported by or contradicting the given documents.

Reference Documents:
{documents}

Answer to Evaluate:
{generation}

Evaluation Guidelines:
1. Carefully read both the reference documents and the answer.
2. Identify all factual claims made in the answer.
3. For each claim, determine if it is:
   a) Fully supported by the documents
   b) Partially supported with some extrapolation
   c) Not mentioned in the documents
   d) Contradictory to the information in the documents
4. Consider the overall context and implied information.
5. Be strict in your evaluation; any unsupported or contradictory information should be flagged.

Provide your evaluation as a JSON object with the following structure:
{{
    "score": ["yes" if fully factual, "no" if any hallucination detected],
    "confidence": [0-1 scale, e.g., 0.9],
    "unsupported_claims": ["list", "of", "unsupported", "or", "contradictory", "claims"],
    "explanation": "Brief explanation of your decision"
}}

Ensure your evaluation is thorough and objective.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
