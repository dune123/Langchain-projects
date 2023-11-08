from langchain import HuggingFaceHub
from langchain import PromptTemplate,LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import streamlit as st 
from dotenv import load_dotenv
import os

load_dotenv()
def get_hugging_face(question):  
    llm_huggingface=HuggingFaceHub(repo_id="tiiuae/falcon-7b",model_kwargs={"temperature":0.6,"max_length":64})
    response=llm_huggingface(question)
    return response

st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input=st.text_input("Input: ",key="input")
response=get_hugging_face(input)

submit=st.button("Ask a question")

if submit:
    st.subheader("The response is")
    st.write(response)