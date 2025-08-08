import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import ChatbotFunctions
import os
from langchain.schema.output_parser import StrOutputParser
import numpy
load_dotenv('secrets.env')

PINECONE_APIKEY = os.getenv('PINECONE_APIKEY')
OPEN_AI_APIKEY = os.getenv('OPEN_AI_APIKEY')

pc = Pinecone(api_key=PINECONE_APIKEY)
vectors = pc.Index("qmuldocs")
Embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = PineconeVectorStore(index=vectors,embedding=Embedder)

llmodel = ChatOpenAI(model_name = 'gpt-4o', api_key=OPEN_AI_APIKEY)

st.title('QMUL EECS ChatBot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message,dict):
        with st.chat_message(message['role']):
            st.markdown(message['content'])

if usermsg := st.chat_input('type a message'):
    st.session_state.messages.append({'role': 'user', 'content':usermsg})
    with st.chat_message('user'):
        st.markdown(usermsg)
        chat_history = 'user: ' + usermsg + '\n'
        print(usermsg)

    with st.chat_message('assistant'):
        strem = st.empty()
        full_msg = ''
        context = ChatbotFunctions.retrieval(vector_store=vectors,question=usermsg, knn = 10)
        temp =  "please answer the following question {question} using the information {context} only. You are an assistant in Queen mary's university of london and you are answering a student, if there is no context then ask user to elaborate on their question without using context,relate it to queen mary's EECS department. write within 100 words and use a friendly demeanor."

        prompt = ChatPromptTemplate.from_template(temp)
        chain = prompt | llmodel | StrOutputParser()
        for chunk in chain.stream({"context":context, "question":usermsg}):
            full_msg += chunk
            strem.markdown(full_msg)
        st.session_state.messages.append({'role': 'assistant', 'content':full_msg})
        chat_history = 'Assistant: ' + full_msg + '\n'
