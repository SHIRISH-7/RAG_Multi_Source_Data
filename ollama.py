
# creating bot
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

#loading env variables
# os.environ['OPENAI_API_KEY']=os.getenv("open_ai_key")
os.environ['LANGCHAIN_API_KEY']=os.getenv("langchain_api")
os.environ['LANGCHAIN_TRACING_V2']="true"

# creating bot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please provide the response to the queries of the user."),
        ("user","Question: {question}")
    ]
)

#creating framework
st.title('Langchain with Llama3 API')
input_text=st.text_input('Search the topic you want')

#calling Open AI
llm=Ollama(model="llama3")
# llm=ChatOpenAI(model='gpt-3.5-turbo')
output_parser=StrOutputParser()

#building chains
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
