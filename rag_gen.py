import os

from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def pdf_loader(pdf_file):
    loader = PyPDFLoader(pdf_file)
    return loader.load()

def split_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                              chunk_overlap=30)
    return splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings

def embed_texts(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    embeddings = embedding_model.embed_documents(texts)
    return embeddings, texts

def vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vdb = FAISS.from_documents(chunks,embedding_model )
    return vdb

def rag(vdb, question):
    retriever = vdb.as_retriever()
    llm = ChatGroq(api_key=GROQ_API_KEY, model='llama-3.3-70b-versatile' ) 
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents = True
    )

    return rag_chain({'query':question})

def main():
    print('Extraction à partir d\'un PDF')
    documents = pdf_loader('knowledge_docs/formation_IA.pdf')

    print("Chunking du text")
    chunks = split_document(documents)


    print("Créer Vector DB (FAISS) et la remplir")
    vdb = vector_db(chunks)

    print('Utiliser le RAG : ')
    question = 'Could you give me the content of the last session of the training'

    results = rag(vdb, question)

    print('AI :', results['result'])
    for doc in results['source_documents']:
        print(doc.metadata['source'])

if __name__ == "__main__":
    main()