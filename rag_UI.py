import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_OPTIONS = {
    "LLaMa 3.3 (70B)": "llama-3.3-70b-versatile",
    "Gemma 2 (9B)": 'gemma2-9b-it',
    "LLaMa 3 (8B)": 'llama3-8b-8192',
    'qwen': 'qwen-qwq-32b'
}

def llm_answer(history, model):
    client = Groq( api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages= history,
        model=model,
    )
    return chat_completion.choices[0].message.content

st.set_page_config(page_title="Chatbot TEK-UP", layout='centered')
st.title("📚 TEK-UP PDF Chatbot")

# Model selection
selected_model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_model_label]

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = [{
        'role': 'system',
        'content': 'You are Thabet, TEK-UP university assistant. Answer every question in a fun and loud tone!'
    }]

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF to chat with", type=["pdf"])

# Initialize vector DB
if uploaded_pdf and 'vdb' not in st.session_state:
    with st.spinner("Lecture du PDF..."):
        
        # Définir le dossier d'upload
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Définir le chemin complet pour enregistrer le fichier
        pdf_path = os.path.join(upload_dir, uploaded_pdf.name)  

        # Enregistrer le fichier PDF
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        # Chargement du PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Chunking du PDF
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
        chunks = splitter.split_documents(documents)

        # Insertion dans la BDD vectorielle
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        vdb = FAISS.from_documents(chunks, embedding_model)

        st.session_state.vdb = vdb
    st.success("PDF uploadé avec succes")

# Chat
user_input = st.chat_input("Posez une question à propos du pdf..." if uploaded_pdf else "Vous devez uploader un PDF!")

# Handle user input
if user_input and uploaded_pdf:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.spinner("Je réfléchis..."):
        retriever = st.session_state.vdb.as_retriever()
        llm = ChatGroq(api_key=GROQ_API_KEY, model=selected_model)

        relevant_docs = retriever.get_relevant_documents(user_input)

        context = "\n\n".join(relevant_doc.page_content for relevant_doc in relevant_docs)

        query = f"""
         "You are Thabet, TEK-UP university assistant. Answer the question below using the provided context.\n\n"
          f"Context:\n{context}\n\n"
         f"Question: {user_input}\n\n"
         "Answer in a loud and fun tone:"
        """


        result = llm.invoke(query)

        answer = result.content


        # rag_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     retriever=retriever,
        #     return_source_documents=True
        # )
        # result = rag_chain({'query': user_input})
        # answer = result['result']
        
    
        
    st.session_state.history.append({"role": "assistant", "content": answer})

# Display chat
for message in st.session_state.history:
    if message['role'] != 'system':
        with st.chat_message(message['role']):
            st.markdown(message['content'])

     