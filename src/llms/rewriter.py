### Question Re-writer

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import llm

# Prompt
re_write_prompt = PromptTemplate(
    template="""You are a question re-writer. Your task is to convert an input question into a better version that is optimized for vectorstore retrieval. 
    Please take the initial question and formulate an improved version. 
    Here is the initial question: {question}. 
    Provide the improved question without any preamble: """,
    input_variables=["question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
