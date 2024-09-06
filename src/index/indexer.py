import traceback
from datetime import datetime

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

from .clickup import get_clickup_docs

CHROMA_PATH = "chroma"

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OllamaEmbeddings(model="llama3.1", ),
)

retriever = db.as_retriever()


def ingest_document(
    click_up_url: str,
):
    start_time = datetime.now()
    try:
        workspace_id, doc_ids = click_up_url.split("app.clickup.com")[1].split("/v/dc/")
        ids = doc_ids.split("/")
        doc_id, sub_doc_id = '', ''
        if len(ids) == 2:
            doc_id, sub_doc_id = ids
        if len(ids) == 1:
            doc_id = ids[0]
        chunks = load_document(workspace_id, doc_id, sub_doc_id)
        add_to_chroma(chunks)
        end_time = datetime.now()
        print(f"⏰ Time taken to load repo in Chroma: {end_time - start_time}")
    except ValueError as e:
        print("❌ Error occurred", e)
        traceback.print_exc()
        exit(1)


def load_document(workspace_id, doc_id, sub_doc_id):
    documents = get_clickup_docs(workspace_id, doc_id, sub_doc_id)
    if not documents:
        print(f"❌ No documents found for {workspace_id}/{doc_id}/{sub_doc_id}")
        return []

    print(f"📝 ClickUp Doc {workspace_id}/{doc_id}/{sub_doc_id} loaded.")
    return split_documents(
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


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2048,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="llama3.1", )
    )

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


def calculate_chunk_ids(chunks):
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
