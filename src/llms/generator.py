### Generate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import llm

prompt = PromptTemplate(
    template="""You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to provide a comprehensive and accurate answer to the question. 
    If you don't know the answer, just say that you don't know. Provide your explanation with full confidence in markdown, give a text response. Make sure user is fulfilled with you answer.
    Question: {question} 
    Context: {context} 
    Answer:
    Return the binary answer as a JSON object with a single key 'answer' and no additional keys.
    """,
    input_variables=["question", "context"],
)

rag_chain = prompt | llm | StrOutputParser()
