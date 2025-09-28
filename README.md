# 🤖 GenAI Chatbot with RAG and Streamlit

An intelligent chatbot powered by **Groq LLMs (LLaMa, Gemma, Qwen)** with two key features:
- **Interactive Chatbot**: Chat with multiple models using a simple CLI or a user-friendly **Streamlit UI**.
- **RAG (Retrieval Augmented Generation)**: Upload a PDF, and the chatbot answers your questions based on its content using **LangChain + FAISS + HuggingFace embeddings**.

This project showcases practical applications of **Generative AI, NLP, and Retrieval systems**.

## ✨ Features
- 🔹 Multi-model chatbot (LLaMa 3.3, Gemma 2, Qwen, etc.)
- 🔹 Streamlit UI with conversational memory
- 🔹 Retrieval-Augmented Generation (RAG) using PDF knowledge
- 🔹 Vector database with **FAISS** for fast similarity search
- 🔹 Secure API key management with `.env`


## 📂 Project Structure
GenAI_Chatbot/
│── chatbot.py              # Simple CLI chatbot
│── chatbotUI.py            # Streamlit-based chatbot
│── genAI-chatbot.ipynb     # Notebook demo with LLM
│── LLM_using_API.ipynb     # Notebook demo (LLM API usage)
│── rag_gen.py              # RAG pipeline with FAISS + LangChain
│── rag_UI.py               # Streamlit RAG chatbot with PDF upload
│── requirements.txt        # Dependencies
│── .env                    # API keys (not uploaded to GitHub)
│── .gitignore              # Ignore sensitive files
│── uploads/                # Uploaded PDFs
│── knowledge_docs/         # Sample PDFs
