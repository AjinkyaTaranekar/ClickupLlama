from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an expert evaluator tasked with assessing the quality and usefulness of an answer in addressing a given question. Your goal is to determine if the answer effectively resolves the user's query.

Question: {question}

Answer to Evaluate:
{generation}

Evaluation Guidelines:
1. Carefully read both the question and the provided answer.
2. Assess the answer based on the following criteria:
   a) Relevance: Does the answer directly address the question?
   b) Completeness: Does it cover all aspects of the question?
   c) Clarity: Is the answer easy to understand?
   d) Accuracy: Does the answer seem factually correct? (Note: You're not fact-checking, but assessing apparent accuracy)
   e) Conciseness: Is the answer appropriately detailed without unnecessary information?
3. Consider whether the answer would satisfy the user's information need.
4. If the answer is partially useful but lacks critical information, it should be considered insufficient.

Provide your evaluation as a JSON object with the following structure:
{{
    "score": ["yes" if the answer is sufficient, "no" if insufficient],
    "confidence": [0-1 scale, e.g., 0.8],
    "strengths": ["list", "of", "answer's", "strong", "points"],
    "weaknesses": ["list", "of", "answer's", "weak", "points"],
    "suggestion": "Brief suggestion for improvement if the answer is insufficient"
}}

Ensure your evaluation is thorough, fair, and focused on the user's needs.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
