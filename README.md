# **👑 Victoria: The Victorian Era Histographer**

---

Victoria is an advanced AI-driven research assistant designed to explore the Industrial Revolution through the eyes of its most iconic figures. By combining Retrieval-Augmented Generation (RAG) with dynamic persona modeling, Victoria transforms static historical archives into an immersive, conversational experience.

## Main Function

---

The primary goal of Victoria is to provide a grounded historical research tool where every claim is backed by primary and secondary sources from the data folder documents. For better user experience, users can consult with different historical personalities, each with varying "Character Strengths," to retrieve data on the Industrial Revolution, Victorian currency, and industrial statistics or any other information regarding Victorian Era (1837 - 1901).

## Techniques & Technology Stack

---

- LLM Orchestration: LangChain for agentic reasoning and tool binding.

- RAG (Retrieval-Augmented Generation): Custom retrieval pipeline using ChromaDB to query a library of historical PDFs.

- Dynamic Personas: Context-aware system prompting that adjusts temperature and vocabulary based on user selection.

- Frontend: Streamlit for a responsive, state-managed web interface.

- Tools: Custom Python tools for currency conversion and mathematical industrial data analysis.

## Project Structure

---

VICTORIA/
├── core/
│   ├── retriever.py      # Logic for ChromaDB connection and document retrieval
│   └── tools.py          # Custom LangChain tools (Currency, Stats)
├── data/
│   ├── chroma_db/        # Persistent vector store index
│   └── [PDFs]            # Primary sources (Sadler Report, Mines Act, etc.)
├── app.py                # Main Streamlit application and Agent logic
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (OpenAI API Key)
└── .gitignore            # Files excluded from version control

## Project Architecture

---

### Section 0: Imports

In this section, I load the essential libraries required to run the engine. I include Streamlit for the web interface, OpenAI for the language model, and LangChain to coordinate the agent's "thinking" and "tool use".

### Section 1: Core Configuration & Constants

This serves as my app's "Dictionary". I map technical PDF filenames to readable titles so that citations look professional, and I assign specific emojis to each character to maintain a consistent visual style.

### Section 2: State Management

This is the "Short-Term Memory" of my application. Since Streamlit re-runs the entire script with every click, I use this section to ensure that your conversation history and selected settings are preserved so the app doesn't "forget" the context.

### Section 3: Helper Functions & Tools

Here, I define the specialized "Skills" given to the AI:

Theme Identifier: I use this to summarize your queries into two-word topics for history tracking.

search_royal_archives: This is the core Retrieval tool. It connects to the retriever to fetch factual data from my library and returns page numbers for accuracy.

### Section 4: Agent Logic

This is the "Brain" where I create the persona. I combine the character’s personality with the Character Strength slider. I use this logic to change the AI's "temperature"—making it either a strict academic or a theatrical performer.

### Section 5: Sidebar & Navigation

I use this section to handle the "Control Panel". I provide the interface for switching characters and a clickable history of past inquiries, using unique keys to prevent technical errors when titles are similar.

### Section 6: Main Chat Interface

This section handles the "Theater" of the app. I display the messages and ensure the icon (like the Crown or Knife) stays locked to the specific message it was sent with, even if you switch characters later.

### Section 7: Chat Input & Execution

This is the "Engine Room" where I process the work. I capture your input, display the "Searching Royal Archives..." status, run the agent, and then save the final answer and evidence back into the history.

### retriever.py

I use this file to manage the Vector Database (ChromaDB). It is responsible for taking your PDF documents, breaking them into searchable "chunks," and finding the most relevant pieces of information when the agent asks a question.

### tools.py

I store my custom-built historical tools here. This includes the Victorian Currency Converter and the Industry Stats Calculator. These allow the AI to perform precise mathematical tasks that a standard language model might struggle with, such as converting old British Pounds or calculating factory growth rates.

## License

---

This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: