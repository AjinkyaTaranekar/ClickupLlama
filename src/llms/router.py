### Router

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an expert at determining the appropriate data source for a given user question. \n
    If the question pertains to LLM agents, prompt engineering, or adversarial attacks, route it to the vectorstore. \n
    For all other topics, route the question to web search. \n
    Provide a binary choice: 'web_search' or 'vectorstore' based on the question. \n
    Return a JSON object with a single key 'datasource' and no preamble or explanation. \n
    Question to route: {question}""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()
