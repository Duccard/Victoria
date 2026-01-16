import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA

# 1. Setup environment
load_dotenv()


def ask_victoria(query: str) -> str:
    """
    Connects to the database and uses an LLM to answer a question
    based on the ingested Victorian documents.
    """
    # 2. Load the database (The Librarian)
    db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())

    # 3. Initialize the LLM (The Brain)
    # We use 'gpt-4o-mini' for high speed and low cost
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 4. Create the Chain
    # This automatically retrieves relevant docs and passes them to the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),  # Increased from 3 to 6
    )

    # 5. Get the answer
    response = qa_chain.invoke(query)
    return response["result"]


if __name__ == "__main__":
    print("--- Victoria Chat System Active ---")

    # This 'input()' function pauses the script and waits for your keyboard
    user_question = input("Enter your question for Victoria: ")

    print(f"\n--- Searching the archives... ---")
    answer = ask_victoria(user_question)
    print(f"\nVictoria: {answer}\n")
