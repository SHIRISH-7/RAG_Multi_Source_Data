#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent,AgentExecutor
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("open_ai_key")

# In[2]:


#loading the text doc
txt_loader=TextLoader('chardham.txt').load()

# In[3]:


# splitiing the documents
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk_docs=[]
chunk_docs.extend(text_splitter.split_documents(txt_loader))

# In[4]:


#creating embeddings
db=FAISS.from_documents(chunk_docs,OpenAIEmbeddings())

# In[27]:


#chat prompt template
prompt="""You are a helpful assistant which answers only based upon context.Answer the following question based upon the context only.
    Think step by step before providing a detailed answer.
    Give all the related data which are present in the documents for the particular query.
    To do this frst translate the language of input to hindi then search in the context then again translate the response in 
    the language of input
    For each extra details i will give you $50,000 for the correct ones.
    I will tip you $10,000 if the user finds the answer helpful.
    Translate the response in the language of input.
    Do not answer the queries which are not related to context.
    """

# In[19]:


llm=ChatOpenAI(model='gpt-4-turbo',temperature=0)
# from langchain_community.llms import Ollama
# #Ollama llam3
# llm=Ollama(model='llama3')
# llm

# In[20]:


#creating retreivar
#retreivar acts as an additional step in between db vector search and the llm it will return the document for a particular query
retreiver=db.as_retriever()

# In[28]:


open_ai_prompt = hub.pull("hwchase17/openai-tools-agent")
open_ai_prompt.messages[0]=SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompt))

# In[29]:


retrieval_tool_doc=create_retriever_tool(retreiver,'Chardham_Information','It will search the  different contact information of \
    hospital,police station,petrol pump,taxi rates,helicopter rates,cart rates etc. which is present in the documents related to chardham. for any information related to chardhmam \
       use this tool. ')


# In[30]:



def search_answer(query):
    refined_prompt="""You are a ChardhamGPT which helps user as a guide during yatra.
        You can also create a travel plan for chardham,along with all the details should be taken care of etc. from 
        your training knowledge.
        You know only about Chardham yatra.
        Answer the following question based on the provided context.
        If there is no information in context then answer according to your training knowledge which are related to ChardhamYatra.
        If you answer the queries which are not related to Chardham Yatra I will fine you $1,000,000.
        Think step by step before providing a detailed answer.
        I will tip you $10,000 if the user finds the answer helpful.
        If there is relatable information in context then identify and transform the context in the language of the human/user/query.
        Do not answer the queries which are not related to chardham yatra.
        <context>
        {context}
        </context>
        """
    tools=[retrieval_tool_doc] 
    agent=create_openai_tools_agent(llm,tools,open_ai_prompt)
    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=False)
    response=agent_executor.invoke({"input":query})

    refine=ChatPromptTemplate.from_messages(
    [
        ("system",refined_prompt),
        ("human","The language of the answer should be same as of query.You have to include atleast all  \
        information which are present in system prompt.Question: <query>{query}</query>Answer:")
    ])

    chain=refine|llm
    answer=chain.invoke({"query":query,"context":response})
    return answer.content


# In[33]:


# search_answer('কেদারনাথে হেলিকপ্টার রেট')#helicopter rate in kedarnath
# Helicopter rates in Kedarnath are as follows:\n\n1. **From Guptakashi:**\n - Operators: Aero Aircraft, Aran Aviation\n - Round Trip Fare: 
# ₹7,750.00\n\n2. **From Fata:**\n - Operators: Pawan Hans, Chipsan Aviation, Thambi Aviation, Pinnacle Air\n - Round Trip Fare: ₹4,720.00
# \n\n3. **From Sersi:**\n - Operators: Aero Aircraft, Himalayan Heli, Kestrel Aviation\n - Round Trip Fare: ₹4,680.00\n\n
# These helicopter services make Kedarnath Yatra more convenient and faster.

# In[34]:


# search_answer('mujhe kedarnath mai helicopteer lena hai')

# In[ ]:


print(search_answer(input('Pleae write your query related to chardham.')))
