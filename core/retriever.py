import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever

load_dotenv()

CHROMA_PATH = "chroma_db"


def get_retriever() -> BaseRetriever:
    """
    Initializes a MultiQueryRetriever using the persisted Chroma database.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": 3}), llm=llm
    )
    return retriever
