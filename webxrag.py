import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
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
from llama_index.legacy import (StorageContext, load_index_from_storage)
import openai
import nest_asyncio

nest_asyncio.apply()
openai.api_key = "YOUR_OPEN_AI_KEY"
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4-0125-preview", temperature=0)

DEFAULT_CONTEXT_PROMPT_TEMPLATE_WEB = """
You are an artificial intelligence assistant designed to help answer questions related to I-Venture at ISB or DLabs ISB. 
When the context does not have the necessary information, search the web and provide a detailed answer with logical paragraphs. 
Make sure to list the sources of your answer.
"""

DEFAULT_CONTEXT_PROMPT_TEMPLATE_1 = """
 You're an AI assistant designed to help answer questions related to I-Venture at ISB or DLabs ISB.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:
 {context_str}
 Instruction: Based on the above context, provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
"""

condense_prompt = (
    "Given the following conversation between a user and an AI assistant and a follow-up question from the user,"
    "rephrase the follow-up question to be a standalone question.\n"
    "Chat History:\n"
    "{chat_history}"
    "\nFollow-Up Input: {question}"
    "\nStandalone question:"
)

def indexgenerator(indexPath, documentsPath):
    # check if storage already exists
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    if not os.path.exists(indexPath):
 #       st.write("Creating new index...")
        # load the documents and create the index
        entity_extractor = EntityExtractor(prediction_threshold=0.2, label_entities=False, device="cpu")
        node_parser = SentenceSplitter(chunk_overlap=102, chunk_size=1024)
        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()

        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0), embed_model=embed_model)
        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        # load existing index
#        with st.spinner("Loading existing index..."):
            storage_context = StorageContext.from_defaults(persist_dir=indexPath)
            index = load_index_from_storage(storage_context, service_context=ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0), embed_model=embed_model))

    return index

indexPath = '/home/ubuntu/Indices/dlabs'
documentsPath = '/home/ubuntu/'
index = indexgenerator(indexPath, documentsPath)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
topk = 2
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=topk)
postprocessor = LongContextReorder()

def chat():
    bm25_flag = True

    try:
        bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=topk)
    except:
        source_nodes = index.docstore.docs.values()
        nodes = list(source_nodes)
        bm25_flag = False

    class HybridRetriever(BaseRetriever):
        def __init__(self, vector_retriever, bm25_retriever):
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            super().__init__()

        def _retrieve(self, query, **kwargs):
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
            all_nodes = bm25_nodes + vector_nodes
            query = str(query)
            all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes, query_bundle=QueryBundle(query_str=query.lower()))
            return all_nodes[0:topk]

    if bm25_flag:
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    else:
        hybrid_retriever = vector_retriever

    query_engine = RetrieverQueryEngine.from_args(retriever=hybrid_retriever, service_context=service_context, verbose=True)

    client = openai.OpenAI(api_key="YOUR_PERPLEXITY_KEY", base_url="https://api.perplexity.ai")
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [{"role": "assistant", "content": "Ask me any question"}]
    if "message_history" not in st.session_state.keys():
        st.session_state.message_history = [ChatMessage(role=MessageRole.ASSISTANT, content="Ask me any question")]

    if user_input := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": str(user_input)})
    
    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
#        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_input = st.session_state.messages[-1]["content"]
                st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=user_input))

                response_context = CondensePlusContextChatEngine.from_defaults(
                    query_engine,
                    context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1,
                    condense_prompt=condense_prompt,
                    chat_history=st.session_state.message_history
                ).chat(str(user_input))

                response_content = response_context.response if hasattr(response_context, 'response') else str(response_context)

                st.write(response_content)
                st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=str(response_content)))

                if st.button("Get enhanced answer"):
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": user_input,
                            },
                        ]

                        modified_messages = []
                        for message in messages:
                            if message["role"] == "user":
                                modified_message = {"role": "user", "content": f"{message['content']} I-venture ISB"}
                            else:
                                modified_message = message
                            modified_messages.append(modified_message)

                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = client.chat.completions.create(
                                    model="llama-3-sonar-large-32k-online",
                                    messages=modified_messages,
                                )
                                web_content = response.choices[0].message.content

                                enhanced_response = CondensePlusContextChatEngine.from_defaults(
                                    query_engine,
                                    context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_WEB,
                                    condense_prompt=condense_prompt,
                                    chat_history=st.session_state.message_history
                                ).chat(f"{user_input} {web_content}")
                                
                                enhanced_response_content = enhanced_response.response if hasattr(enhanced_response, 'response') else str(enhanced_response)

                                st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=user_input))
                                st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=enhanced_response_content))

                                st.write("Enhanced Answer: ")
                                st.write(enhanced_response_content)

                    except Exception as e:
                        st.write(f"An error occurred: {e}")

def main():
    st.title("Dlabs Chatbot")
    chat()

if __name__ == "__main__":
    main()
