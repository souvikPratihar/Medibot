# MediBot — Medical Chatbot with RAG (Streamlit + LangChain + FAISS)

MediBot is a lightweight, privacy‑friendly medical Q&A assistant built with Streamlit and LangChain. It uses a local FAISS vector store of medical PDFs and HuggingFace sentence embeddings to retrieve relevant context, then answers queries with an LLM (served via Groq’s OpenAI‑compatible API). Deployable on Streamlit Cloud.

Important: MediBot is an educational assistant, not a substitute for professional medical advice.

## Features
- Retrieval‑Augmented Generation (RAG) over your own PDFs.
- Local FAISS index for fast semantic search.
- SentenceTransformers embeddings: all‑MiniLM‑L6‑v2.
- Groq LLM endpoint (OpenAI‑compatible) with llama3‑8b‑8192 by default.
- Clean Streamlit chat UI with conversation history.
- Incremental vectorstore updates (only new PDFs processed).
- Configurable prompt with strict “answer from context only”.

## Tech stack
- Python, Streamlit
- LangChain, FAISS
- sentence-transformers
- HuggingFace Hub (embeddings)
- Groq API (OpenAI‑compatible Chat API)

## Project structure
- app.py — Streamlit app (entry point)
- create_memory_for_llm.py — builds/updates the FAISS index from PDFs
- connect_memory_with_llm.py — CLI testing against the vector store
- data/ — place source PDFs here
- vectorstore/
  - db_faiss/ — FAISS index files
  - processed_files.log — tracks which PDFs are indexed
- requirements.txt — runtime dependencies
- Pipfile / Pipfile.lock — optional (pipenv)
- .gitignore — excludes secrets and local caches
- README.md — this file

## Setup (local)
1) Create and activate a virtual environment
- navigate to the project folder
- pipenv shell

2) Install dependencies
- pipenv install -r requirements.txt

3) Add PDFs
- Put medical PDFs into the data/ folder.

4) Build the vector store
- python create_memory_for_llm.py
- This creates vectorstore/db_faiss and logs processed files.

5) Set environment variables (local)
Create a .env file with:
- GROQ_API_KEY

6) Run the app
- streamlit run app.py

## Deployment (Streamlit Cloud)
1) Push repo (including vectorstore/) to GitHub. Do not upload .env.
2) In Streamlit Cloud:
- Create a new app, select your repo and app.py as the entry file.
- Go to Settings → Secrets and add:
  - GROQ_API_KEY = "value"
  
Streamlit injects them as environment variables.

3) Deploy.

## Usage
- Open the app.
- Type a medical question (e.g., “What are early symptoms of lungs cancer?”).
- MediBot retrieves the most relevant chunks from your PDFs and generates an answer strictly from that context.
- If context is missing, it will say it doesn’t know.

## Updating the knowledge base
- Drop new PDFs into data/.
- Run: python create_memory_for_llm.py
- Only new files will be embedded and added to FAISS. Commit the updated vectorstore.

## Configuration notes
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store path: vectorstore/db_faiss
- LLM: llama3‑8b‑8192 via Groq 
- Retrieval: top-k=3

## Roadmap ideas
- Source citation display (page numbers, doc titles).
- File uploader to build vectorstore in the cloud (guarded).
- Feedback buttons to refine answers.
- Multi‑model routing and cost logging.

## Disclaimer
MediBot is for educational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.
