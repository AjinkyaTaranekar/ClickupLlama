### Hallucination Grader

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

# Prompt
prompt = PromptTemplate(
    template="""You are an expert tasked with evaluating the factual accuracy of an answer based on provided documents. \n 
    Below are the documents containing the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    And here is the answer: {generation}
    Please determine if the answer is factually accurate and supported by the documents. Provide a binary score 'yes' or 'no'. \n
    Return the score as a JSON object with a single key 'score' and no additional commentary or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
