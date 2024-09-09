import traceback
from datetime import datetime
from typing import List, Optional

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

from .clickup import get_clickup_docs

CHROMA_PATH = "chroma"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100


def get_chroma_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="llama3.1"),
    )


db = get_chroma_db()
retriever = db.as_retriever()


def ingest_document(click_up_url: str) -> None:
    start_time = datetime.now()
    try:
        workspace_id, doc_ids = parse_clickup_url(click_up_url)
        chunks = load_document(
            workspace_id, doc_ids["doc_id"], doc_ids.get("sub_doc_id")
        )
        if chunks:
            add_to_chroma(chunks)
            end_time = datetime.now()
            print(f"⏰ Time taken to load repo in Chroma: {end_time - start_time}")
        else:
            print("❌ No chunks were generated from the document.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        traceback.print_exc()


def parse_clickup_url(url: str) -> tuple[str, dict]:
    try:
        parts = url.split("app.clickup.com")[1].split("/v/dc/")
        workspace_id = parts[0]
        ids = parts[1].split("/")
        return workspace_id, {
            "doc_id": ids[0],
            "sub_doc_id": ids[1] if len(ids) > 1 else None,
        }
    except IndexError:
        raise ValueError("Invalid ClickUp URL format")


def load_document(
    workspace_id: str, doc_id: str, sub_doc_id: Optional[str]
) -> List[Document]:
    documents = get_clickup_docs(workspace_id, doc_id, sub_doc_id or "")
    if not documents:
        print(f"❌ No documents found for {workspace_id}/{doc_id}/{sub_doc_id or ''}")
        return []

    print(f"📝 ClickUp Doc {workspace_id}/{doc_id}/{sub_doc_id or ''} loaded.")
    return split_documents(
        [
            Document(
                page_content=f"#{doc.get('name', '')}\n\n\n{doc.get('content', '')}",
                metadata={
                    "file_path": f"{workspace_id}/{doc_id}/{doc.get('id', '')}",
                    "workspace_id": workspace_id,
                    "doc_id": doc_id,
                    "sub_doc_id": doc.get("id", ""),
                },
            )
            for doc in documents
        ]
    )


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: List[Document]) -> None:
    db = get_chroma_db()
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"📄 Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ New documents added")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
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
