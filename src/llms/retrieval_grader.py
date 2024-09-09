from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an expert evaluator tasked with assessing the relevance of retrieved documents to a user's question. Your goal is to determine whether a document contains useful information for answering the question, even if it doesn't provide a complete answer.

User's Question: {question}

Retrieved Document:
{document}

Evaluation Guidelines:
1. Carefully read both the user's question and the retrieved document.
2. Look for the following indicators of relevance:
   a) Keywords or phrases from the question appearing in the document
   b) Concepts or ideas related to the question's topic
   c) Information that could contribute to forming an answer, even if partial
   d) Context that helps understand the question's subject matter
3. Consider the document relevant if it provides any useful information, even if it's not a perfect match.
4. Be somewhat lenient in your evaluation; the primary goal is to filter out clearly irrelevant documents.
5. If in doubt, lean towards marking the document as relevant.

Scoring:
- Score 'yes' if the document is relevant or potentially useful.
- Score 'no' if the document is clearly unrelated or contains no useful information.

Provide your evaluation as a JSON object with the following structure:
{{
    "score": ["yes" or "no"],
    "confidence": [0-1 scale, e.g., 0.8],
    "key_matches": ["list", "of", "key", "terms", "or", "concepts", "found"]
}}

The 'confidence' field should reflect how certain you are about your relevance decision.
The 'key_matches' field should list important terms or concepts from the question found in the document.

Remember, no additional explanation is needed beyond this JSON object.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
