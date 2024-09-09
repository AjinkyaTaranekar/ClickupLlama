from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import llm

re_write_prompt = PromptTemplate(
    template="""You are an expert question optimizer for vectorstore retrieval. Your task is to refine the given question to improve its effectiveness in retrieving relevant information. Follow these guidelines:

1. Identify and retain the core intent of the original question.
2. Expand on any acronyms or domain-specific terms.
3. Add relevant context if it's implied but not explicitly stated.
4. Use specific, descriptive language to enhance searchability.
5. Break down complex questions into simpler, more focused queries if necessary.
6. Ensure the rewritten question is clear, concise, and unambiguous.

Original question: {question}

Rewritten question:""",
    input_variables=["question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
