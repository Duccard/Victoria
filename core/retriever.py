import os
from typing import List, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables (API Keys)
load_dotenv()

# Setup Paths
DATA_PATH: str = "data/"
CHROMA_PATH: str = "chroma_db/"


def ingest_docs() -> Chroma:
    print("--- Starting Ingestion ---")
    documents: List[Document] = []

    # 1. Load PDFs (Updated to ensure Page Numbers are kept)
    if any(f.endswith(".pdf") for f in os.listdir(DATA_PATH)):
        pdf_loader = DirectoryLoader(
            DATA_PATH,
            glob="./*.pdf",
            loader_cls=PyPDFLoader,  # This keeps the page numbers!
            show_progress=True,
        )
        documents.extend(pdf_loader.load())

    # 2. Load CSV
    csv_file_path = os.path.join(DATA_PATH, "key_inventions.csv")
    if os.path.exists(csv_file_path):
        csv_loader = CSVLoader(csv_file_path)
        documents.extend(csv_loader.load())

    # 3. Advanced Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,  # Crucial for pinpointing text location
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Create Vector Store
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    print(f"--- Success: Ingested {len(chunks)} chunks ---")
    return vector_db


def get_retriever() -> BaseRetriever:
    """
    Initializes a MultiQueryRetriever using the persisted Chroma database.

    The MultiQueryRetriever automates the process of prompt tuning by using an LLM
    to generate multiple queries from different perspectives for a given user query.

    Returns:
        BaseRetriever: A LangChain retriever object configured for multi-query retrieval.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Multi-Query Retrieval generates variations of the user's question
    # to find more relevant historical context from the vector store.
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": 3}), llm=llm
    )
    return retriever


if __name__ == "__main__":
    # Ensure the data directory exists before attempting ingestion
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    ingest_docs()
