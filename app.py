import streamlit as st

from config import configure_openai
from search import reconnoiter


def print_chat(role: str, text: str):
    st.chat_message(role).markdown(text)
    st.session_state.messages.append({'role': role, 'text': text})


def submit():
    st.session_state.input = st.session_state.widget
    st.session_state.widget = ''


if 'configured' not in st.session_state:
    configure_openai()
    st.session_state.configured = True
if 'input' not in st.session_state:
    st.session_state.input = ''
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('Car Recommender')
st.text_input('Ask assistant for a car recommendation', key='widget', on_change=submit)
prompt = st.session_state.input

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['text'])

if prompt:
    print_chat('user', prompt)

    response = reconnoiter(prompt)

    print_chat('assistant', response)
