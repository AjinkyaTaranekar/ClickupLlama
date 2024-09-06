### Retrieval Grader

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an evaluator tasked with determining the relevance of a retrieved document to a user's question. \n 
    Below is the retrieved document: \n\n {document} \n\n
    And here is the user's question: {question} \n
    If the document contains keywords or information pertinent to the user's question, mark it as relevant. \n
    The evaluation does not need to be overly strict. The primary objective is to eliminate clearly irrelevant documents. \n
    Provide a binary score 'yes' or 'no' to indicate the document's relevance to the question. \n
    Return the binary score as a JSON object with a single key 'score' and no additional text or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
