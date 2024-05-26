#!/usr/bin/env python
# coding: utf-8

# # RAG With Multi Data Source


# In[1]:


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
#creating agents which will combine the tools,llm and prompt
from langchain.agents import create_openai_tools_agent
#to run the agents we need executer
from langchain.agents import AgentExecutor
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#pulling prompt template from hub
from langchain import hub
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('open_ai_key')
from langchain_community.vectorstores import FAISS

# In[2]:


#wikipedia tool built by langchain
#creating wikipedia api wrapper and running the wrapper with the uery run basically it is kinda json which will go into the api
#in query run and results will get stored in wiki variable
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# In[3]:


#extracting data from documents
docs=PyPDFLoader('keec102.pdf').load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitted_pdfs=text_splitter.split_documents(docs)

# In[4]:


web_doc=WebBaseLoader('https://docs.python.org/3/tutorial/classes.html').load()
splitted_html=text_splitter.split_documents(web_doc)

# In[6]:


db_pdf=FAISS.from_documents(splitted_pdfs,OpenAIEmbeddings())
db_web=FAISS.from_documents(splitted_html,OpenAIEmbeddings())
retreiver_pdf=db_pdf.as_retriever
retreiver_web=db_web.as_retriever

# In[27]:


#pdf and web tool custom made
#this tool is made upon the retreiver first we have to create retreiver then this tool
#it sees and only trigger based on description
retreival_tool_pdf=create_retriever_tool(retreiver_pdf,'economics',"It is economics chapter in which they have talked about indian economy from 1950-1990 with 5 year plan")
retreival_tool_web=create_retriever_tool(retreiver_web,'Python_documentation','It contains documentation for python')

# In[28]:


arxiv_wrapper=ArxivAPIWrapper(top_k_result=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

# In[29]:


#we are defining the series here first it will se the data in website then in pdf then on wkipedia then arxiv
tools=[retreival_tool_web,retreival_tool_pdf,wiki,arxiv]

# In[30]:


llm=ChatOpenAI(model='gpt-3.5-turbo',temperature=0.5)

# In[31]:


prompt=hub.pull('hwchase17/openai-functions-agent')

# In[32]:


agent=create_openai_tools_agent(llm,tools,prompt)

# In[33]:


agent_executer=AgentExecutor(agent=agent,tools=tools,verbose=True)


# In[46]:


agent_executer.invoke({"input":"Tell me about economics"})

# In[ ]:



