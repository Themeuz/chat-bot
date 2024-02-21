import json
import streamlit as st
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

intent = json.load(open('intent.json'))
tags = []
padrao = []
for intent in intent['intent']:
    for padrao in intent['padrao']:
    padrao.append(padrao)
    tags.append(intent['tag'])

vector = TfidfVectorizer()
padrao_scaled = vector.fit_transform(padrao)

Bot = LogisticRegression(max_iter=100000)
Bot.fit(padrao_scaled, tags)

def ChatBot(input_menssage):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intent['intent']:
        if intent['tag'] == pred_tag:
            resposta = random.choice(intent['resposta'])
            return resposta

st.title("lojas AI ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    resposta = f"IA ChatBot: " + ChatBot(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(resposta)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": resposta})