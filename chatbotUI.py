import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


MODEL_OPTIONS = {
    "LLaMa 3.3 (70B)" : "llama-3.3-70b-versatile",
    "Gemma 2 (9B)" : 'gemma2-9b-it',
    "LLaMa 3 (8B)" : 'llama3-8b-8192',
    'qwen' : 'qwen-qwq-32b'
}

def llm_answer(history, model):
    client = Groq( api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages= history,
        model=model,
    )
    return chat_completion.choices[0].message.content

st.set_page_config(page_title = "Chatbot TEK-UP", layout='centered')
st.title("TEK-UP Chatbot")

selected_model_label = st.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))

selected_model = MODEL_OPTIONS[selected_model_label]

if 'history' not in st.session_state:
    st.session_state.history = list()
    st.session_state.history.append(
        {
            'role': 'system',
            'content' : 'You are Thabet, you are TEK-UP unviersity assistant\
            You will answer every queestion while sounding fun and always shouting'
        }
    )

user_input = st.chat_input("Ask me anything : ")

if user_input:
    st.session_state.history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.spinner("I am thinking .. "):
        answer = llm_answer(st.session_state.history, selected_model)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    for message in st.session_state.history:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.markdown(message['content'])