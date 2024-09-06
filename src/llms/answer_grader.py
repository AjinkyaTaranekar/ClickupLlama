### Answer Grader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

# Prompt
prompt = PromptTemplate(
    template="""You are an expert grader tasked with evaluating the usefulness of an answer in addressing a given question. \n 
    Below is the provided answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    And here is the corresponding question: {question}
    Please provide a binary score 'yes' or 'no' to indicate whether the answer effectively resolves the question. \n
    Return the binary score as a JSON object with a single key 'score' and no additional commentary or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
