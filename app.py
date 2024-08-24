import streamlit as st
from streamlit import session_state as state

import config
from ai import AssistantModel
from database import setup_database


def chat_new_message(text: str, role: str):
    state.messages.append({'role': role, 'text': text})
    st.chat_message(role).markdown(text)


if 'configured' not in state:
    config.initialize()
    state.model = AssistantModel(
        vectorstore=setup_database(),
        max_selection=5,
        model_name='gpt-3.5-turbo',
        model_temperature=0.3,
    )
    state.messages = []
    state.configured = True

st.title('Car Recommender')

for message in state.messages:
    st.chat_message(message['role']).markdown(message['text'])

if 'greeted' not in state:
    response = state.model.get_start_message()
    chat_new_message(response, 'assistant')
    state.greeted = True

if prompt := st.chat_input('Ask assistant for a car recommendation'):
    chat_new_message(prompt, 'user')

    with st.spinner("Thinking..."):
        response = state.model.get_response(prompt)

    chat_new_message(response, 'assistant')
