from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

# 1. Path to your PDF folder
DATA_PATH = "/Users/vincas/Turing College Project Files/Victoria/data/"
CHROMA_PATH = "/Users/vincas/Turing College Project Files/Victoria/data/chroma_db"


def build_scholar_db():
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,  # This helps track exactly where text begins
    )

    all_docs = []

    # Process every PDF in the data folder
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Processing {file}...")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            # PyPDFLoader automatically adds 'page' to metadata!
            pages = loader.load_and_split(text_splitter)
            all_docs.extend(pages)

    # 2. Build the Vector Database
    print(f"✨ Creating Vector DB with {len(all_docs)} chunks...")
    db = Chroma.from_documents(
        all_docs, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print("✅ Scholar Database Ready!")


if __name__ == "__main__":
    build_scholar_db()
