from langchain_community.chat_models import ChatOllama

LLM_MODEL = "llama3.1"
llm = (
    ChatOllama(
        model=LLM_MODEL,
        format="json",
        temperature=0,
    ),
)[0]
