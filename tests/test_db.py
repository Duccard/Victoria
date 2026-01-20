import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration Constants
DATA_PATH: str = "data"
CHROMA_PATH: str = "chroma_db"


def load_documents(path: str) -> List[Document]:
    """
    Loads PDF documents from a specified directory.

    Args:
        path (str): The relative path to the folder containing PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects containing
            the text and metadata from the PDFs.
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()


def split_text(documents: List[Document]) -> List[Document]:
    """
    Splits long documents into smaller, manageable chunks for the LLM.

    Args:
        documents (List[Document]): The raw documents loaded from the source.

    Returns:
        List[Document]: A list of smaller Document chunks with overlapping text
             to preserve context between splits.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)


def ingest_docs() -> None:
    """
    The main execution pipeline: loads documents, splits them into chunks,
    and saves them into a local Chroma vector database.

    Note: Requires OPENAI_API_KEY to be set in the .env file.
    """
    load_dotenv()

    print("--- 1. Loading Documents ---")
    raw_docs = load_documents(DATA_PATH)

    print(f"--- 2. Splitting into Chunks (Found {len(raw_docs)} pages) ---")
    chunks = split_text(raw_docs)

    print(
        f"--- 3. Embedding and Saving to {CHROMA_PATH} (Creating {len(chunks)} chunks) ---"
    )

    # Initialize the database and save the chunks
    db = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    print(f"--- Success: Ingested {len(chunks)} chunks into {CHROMA_PATH} ---")


if __name__ == "__main__":
    ingest_docs()
