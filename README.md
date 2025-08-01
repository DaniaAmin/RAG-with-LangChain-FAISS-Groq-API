## RAG Workflow with LangChain, Groq API, and FAISS
🧠 Overview
This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using:

🧱 LangChain for data loading, processing, and chaining

🧠 Groq API as the LLM backend

🧠 FAISS for vector-based semantic search

🌐 BeautifulSoup + WebBaseLoader to scrape content from websites

🎯 Objectives
Use LangChain to load and split data from the web

Convert text into embeddings using Hugging Face models

Store those embeddings in FAISS, a fast vector store

Query the data using RetrievalQA with a Groq-powered LLM

🛠️ Setup Instructions
🔧 Install Required Libraries
bash
Copy
Edit
pip install langchain groq faiss-cpu sentence-transformers beautifulsoup4 langchain-community langchain-groq
✅ Note: groq, faiss-cpu, and langchain-groq are required for LLM and RAG setup.

📂 Project Structure
bash
Copy
Edit
Assignment_1_Agentic_AI.ipynb      # Jupyter notebook with full implementation
README.md                          # Project documentation
🚀 Step-by-Step Workflow
1. 🔍 Load Web Content
python
Copy
Edit
from langchain.document_loaders import WebBaseLoader

url = "https://www.aljazeera.com/where/israel/"  # You can change this
loader = WebBaseLoader(url)
docs = loader.load()
2. ✂️ Split Text into Chunks
python
Copy
Edit
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
3. 🔢 Generate Embeddings and Store in FAISS
python
Copy
Edit
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings_model)
4. 🤖 Load Groq API (LLM)
python
Copy
Edit
import os
from langchain_groq import ChatGroq

os.environ['GROQ_API_KEY'] = 'your_groq_api_key_here'
llm = ChatGroq(model_name="gemma2-9b-it")
5. 📥 Create RetrievalQA Chain
python
Copy
Edit
from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
6. ❓ Ask Questions
python
Copy
Edit
query = "Tell me details of Iran-Israel current conflict?"
answer = qa_chain.run(query)

print(f"Question: {query}\n")
print(f"Answer: {answer}\n")
🧪 Sample Output
vbnet
Copy
Edit
Question: Tell me details of Iran-Israel current conflict?

Answer:
- The conflict is intensifying
- Bombardments and strikes reported
- Preemptive attack by Israel on Iran
- Civilian and military infrastructure targeted
🔒 Notes
Always store API keys securely. Use environment variables or secret management when deploying.

WebBaseLoader uses BeautifulSoup internally to extract clean text from pages.

Embeddings are stored locally in-memory using FAISS.

📚 References
LangChain WebBaseLoader Docs

FAISS Integration in LangChain

Groq API Docs

Hugging Face Embeddings Guide

📌 Author
Dania Amin
Assignment for Agentic AI Project — RAG Pipeline with Groq + LangChain

