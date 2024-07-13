import os
import time

from llama_index.legacy import ServiceContext
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy.schema import QueryBundle
from llama_index.legacy import (StorageContext,load_index_from_storage)
from llama_index.legacy.embeddings import OpenAIEmbedding

import openai
import pandas as pd
import boto3
from rouge import Rouge
from io import StringIO

import streamlit as st

DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context only.
  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above context, provide a detailed answer with logical formation of paragraphs for the user question below.
  Answer "don't know" if information is not present in context. Also, decline to answer questions that are not related to context."
  """

st.set_page_config(page_title="Chat with POM Course Material, powered by AIXplorers", page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ['SECRET_TOKEN']
st.title("Chat with your Course, developed by [GurukulAI](https://www.linkedin.com/in/kush-juvekar/)!! 💬")

with st.sidebar:
    st.title('🤗💬Your Course Made Easy @ Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by [Anupam](https://anupam-purwar.github.io/page/research_group.html)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'

c_option = st.selectbox(
     'Which course would you like to learn today ?',
     ('POM', 'POE', 'CFin','QuantumPhysics','FinTech','Econ101'))

st.write('You selected:', c_option)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question from the course you have selected!!"}
    ]
rouge = Rouge()
try:
  if c_option =='POM':
    indexPath="pom_index"
  elif c_option =='POE':
    indexPath="poe_index"
  elif c_option =='CFin':
    indexPath="cfin_index"
  elif c_option =='QuantumPhysics':
    indexPath="qphy_index"
  elif c_option =='FinTech':
    indexPath="fast_index"
  else:
    indexPath="eco_index"
except:
  indexPath="pom_index"
  
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
storage_context = StorageContext.from_defaults(persist_dir=indexPath)
index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model))
vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
postprocessor = LongContextReorder()
class HybridRetriever(BaseRetriever):
    def __init__(self,vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        all_nodes = bm25_nodes + vector_nodes
        query = str(query)
        all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes,query_bundle=QueryBundle(query_str=query.lower()))
        return all_nodes[0:2]
hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)
query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(query_engine,context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": str(prompt)})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            all_nodes  = hybrid_retriever.retrieve(str(prompt))
            start = time.time()
            response = st.session_state.chat_engine.chat(str(prompt))
            end = time.time()
            st.write(response.response)
            context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes])
            scores=rouge.get_scores(response.response,context_str)
            df = pd.read_csv('logs/course_logs.csv')
            new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"],"Time" : end-start}
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            df.to_csv('logs/course_logs.csv', index=False)
            bucket = 'aiex' # already created on S3
            csv_buffer = StringIO()
            df.to_csv(csv_buffer)
            s3_resource= boto3.resource('s3',aws_access_key_id=os.environ["ACCESS_ID"],aws_secret_access_key=os.environ["ACCESS_KEY"])
            s3_resource.Object(bucket, c_option+'_course_logs.csv').put(Body=csv_buffer.getvalue())
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
