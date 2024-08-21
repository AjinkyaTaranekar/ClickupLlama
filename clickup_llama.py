from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

from clickup import get_clickup_docs

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are a code analysis assistant with deep understanding of software development, capable of interpreting complex codebases.

Given the following context, provide a precise and detailed answer to the question. Ensure your response is clear, concise, and directly addresses the question.

Context:
{context}

---

Question: {question}

Answer:
"""


class ClickUpLlama:
    def __init__(
        self,
        workspace_id: str,
        doc_id: str,
        sub_doc_id: str = "",
        model_name: str = "llama3.1",
    ):
        start_time = datetime.now()
        self.model_name = model_name

        chunks = self.load_document(workspace_id, doc_id, sub_doc_id)
        if chunks:
            self.add_to_chroma(chunks)
        end_time = datetime.now()
        print(f"⏰ Time taken to load repo in Chroma: {end_time - start_time}")

    def load_document(self, workspace_id, doc_id, sub_doc_id):
        documents = get_clickup_docs(workspace_id, doc_id, sub_doc_id)
        if not documents:
            print(f"❌ No documents found for {workspace_id}/{doc_id}/{sub_doc_id}")
            return []

        print(f"🐙 Clickup Doc {workspace_id}/{doc_id}/{sub_doc_id} loaded.")
        return self.split_documents(
            [
                Document(
                    page_content=f"#{doc.get('name', '')}\n\n\n{doc.get('content', '')}",
                    metadata={
                        "file_path": f"{doc.get('workspace_id', '')}/{doc.get('doc_id', '')}/{doc.get('id', '')}"
                    },
                )
                for doc in documents
            ]
        )

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=160,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def get_embedding_function(self):
        return OllamaEmbeddings(model=self.model_name)

    def add_to_chroma(self, chunks: list[Document]):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.get_embedding_function(),
        )

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"📄 Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print("✅ New documents added")
        else:
            print("✅ No new documents to add")

    def calculate_chunk_ids(self, chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("file_path")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def answer_query(self, query_text: str):
        embedding_function = self.get_embedding_function()
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embedding_function
        )

        results = db.similarity_search_with_score(query_text, k=5)
        if not results:
            return {"responses": "No relevant documents found.", "sources": []}

        num_chunks = min(10, len(results))  # Start with the top 10 chunks
        context_texts = [doc.page_content for doc, _score in results[:num_chunks]]

        context_text = "\n\n---\n\n".join(context_texts)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model=self.model_name)
        response_text = model.invoke(prompt)

        if "I don't know" in response_text or len(response_text.strip()) < 10:
            print("🤔 Initial response not confident, refining context...")
            num_chunks += 5  # Increase context size and re-query
            context_texts = [doc.page_content for doc, _score in results[:num_chunks]]
            context_text = "\n\n---\n\n".join(context_texts)

            prompt = prompt_template.format(context=context_text, question=query_text)
            response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results[:num_chunks]]
        return {"responses": response_text, "sources": sources}
