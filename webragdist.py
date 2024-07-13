# import sagemaker
import tiktoken
import streamlit as st
from rouge import Rouge
import os
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import spatial
from ast import literal_eval
from flashrank.Ranker import RerankRequest,Ranker
from llama_index.core import SimpleDirectoryReader
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.schema import MetadataMode
from llama_index.core.schema import QueryBundle
from llama_index.core import (StorageContext,load_index_from_storage)
import openai
from llama_index.llms.groq import Groq
# from openai import OpenAI
import nest_asyncio
from llama_index.llms.perplexity import Perplexity
nest_asyncio.apply()


openai.api_key = ""
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm=OpenAI(model="gpt-3.5-turbo", temperature=0)




DEFAULT_CONTEXT_PROMPT_TEMPLATE_WEB = """
You are an artificial intelligence assistant designed to help answer questions related to I-Venture at ISB or DLabs ISB.
When the context does not have the necessary information, search the web and provide a detailed answer with logical paragraphs.
Make sure to list the sources of your answer.
"""

DEFAULT_CONTEXT_PROMPT_TEMPLATE_1 = """
 You're an AI assistant to help students learn their course material via convertsations.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:


 {context_str}


 Instruction: Based on the above context and web reponse giving more weightage to the web response , provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
 """

condense_prompt = (
  "Given the following conversation between a user and an AI assistant and a follow-up question from the user,"
  "rephrase the follow-up question to be a standalone question.\n"
  "Chat History:\n"
  "{chat_history}"
  "\nFollow-Up Input: {question}"
  "\nStandalone question:"
)


validating_prompt = ("""You are an intelligent bot designed to assist users on an organization's website by answering their queries. You'll be given a user's question and an associated answer. Your task is to determine if the provided answer effectively resolves the query. If the answer is unsatisfactory, return 0.\n
                    Query: {question}  
                    Answer: {answer}
                    Your Feedback:
                    """)



def indexgenerator(indexPath, documentsPath):
    # check if storage already exists
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    if not os.path.exists(indexPath):
        print("Not existing")
        # load the documents and create the index

        entity_extractor = EntityExtractor(prediction_threshold=0.2,label_entities=False, device="cpu") # set device to "cuda" if gpu exists
        node_parser = SentenceSplitter(chunk_overlap = 50, chunk_size = 500)
        transformations = [node_parser, entity_extractor]

        documents = SimpleDirectoryReader(input_dir=documentsPath).load_data()

        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0),embed_model=embed_model)
        index = VectorStoreIndex(nodes, service_context=service_context)

        # store it for later
        index.storage_context.persist(indexPath)
    else:
        #load existing index
        print("Existing")
        storage_context = StorageContext.from_defaults(persist_dir=indexPath)
        index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0),embed_model=embed_model))

    return index


indexPath = r'/home/ubuntu/Indices/dlabs-indices'
documentsPath = ''
index_2000 = indexgenerator(indexPath, documentsPath)



def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances



df=pd.read_csv('/home/ubuntu/embeddings_new.csv',index_col=0)
def create_context(
    question, df=df):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    max_len=1800
    size="ada"
    client =openai.OpenAI(api_key="")
    # Get the embeddings for the question
    # print("question::",question)
    #st.write(question)
    #q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    q_embeddings=client.embeddings.create(input = question, model="text-embedding-ada-002").data[0].embedding
    # Get the distances from the embeddings
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])
    my_dict={}
    for index, element in enumerate(returns):
        my_dict[index] = element
    result_list = [{'id': key, 'text': value} for key, value in my_dict.items()]
    #print(result_list)
    # Return the context
    #return "\n\n###\n\n".join(returns)
    return result_list



def answer_question(question,):
    
 
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    model="gpt-3.5-turbo"
    max_len=1800
    size="ada"
    debug=False
    max_tokens=250
    stop_sequence=None
    context = create_context(question,df=pd.read_csv('/home/ubuntu/embeddings_new.csv',index_col=0))
    print(context)
    ranker=Ranker("ms-marco-MiniLM-L-12-v2",cache_dir="llamaindex_entities_0.2")
    rerankrequest=RerankRequest(query=question,passages=context)
    results=ranker.rerank(rerankrequest)
    print(results)
    text_values = [item['text'] for item in results]
    joined_text = '###'.join(text_values)
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    introduction = 'Use the below text to answer the subsequent question. If the answer cannot be found in the context, write "I could not find an answer."'
    question_ai = f"\n\nQuestion: {question}"
    message = introduction
    #message = message + results + question_ai
    message = message + joined_text + question_ai
    messages = [
        {"role": "system","content": "You are iVBot, an AI based chatbot assistant. You are friendly, proactive, factual and helpful, \
        you answer from the context provided"}, {"role": "user", "content": message},
    ]
    client = openai.OpenAI(api_key="")
    
    try:
        response = client.chat.completions.create(
         model='gpt-3.5-turbo-0125',
         messages=messages,
         temperature=0.01,
         top_p=0.75,
         
        )
      
        ans=response.choices[0].message.content
        # Create a completions using the questin and context
        
        #response = openai.Completion.create(
         #   prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I do not know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
         #   temperature=0.08,
         #   max_tokens=max_tokens,
         #   top_p=0.75,
         #   frequency_penalty=0,
         #   presence_penalty=0,
         #   stop=stop_sequence,
         #   model=model,
        #)
      
        return ans , joined_text #response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


vector_retriever_2000 = VectorIndexRetriever(index=index_2000,similarity_top_k=2)
bm25_retriever_2000 = BM25Retriever.from_defaults(index=index_2000, similarity_top_k=2)
postprocessor = LongContextReorder()

class HybridRetriever(BaseRetriever):
    def __init__(self,vector_retriever_2000, bm25_retriever_2000):
        #self.vector_retriever_1000 = vector_retriever_1000
        #self.bm25_retriever_1000 = bm25_retriever_1000
        self.vector_retriever_2000 = vector_retriever_2000
        self.bm25_retriever_2000 = bm25_retriever_2000
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes_2000 = self.bm25_retriever_2000.retrieve(query, **kwargs)
        vector_nodes_2000 = self.vector_retriever_2000.retrieve(query, **kwargs)
        all_nodes = postprocessor.postprocess_nodes(nodes=bm25_nodes_2000+vector_nodes_2000,query_bundle=QueryBundle(query_str=query.lower()))
        return all_nodes
hybrid_retriever=HybridRetriever(vector_retriever_2000,bm25_retriever_2000)



memory=ChatMemoryBuffer.from_defaults(token_limit=3900) #3900 earlier
rouge = Rouge()

#condense_prompt = (
#  "Given the following conversation between a user and an AI assistant and a follow up question from user,"
#  "rephrase the follow up question to be a standalone question.\n"
#  "Chat History:\n"
#  "{chat_history}"
#  "\nFollow Up Input: {question}"
#  "\nStandalone question:")

#Relaxed prompt with language identification, ans in form of bullet points or short paragraphs



#context_prompt=(
#        "You are a helpful and friendly chatbot who addresses queries in detail regarding I-Venture @ ISB."
#        "Here are the relevant documents for the context:\n"
#        "{context_str}"
#        "\nInstruction: Use the previous chat history above and context, to interact and help the user. Never give any kinds of links, email addresses or contact numbers in the answer."
#        )

#new prompt
import csv
import os

def append_to_csv(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Question', 'Context'])  # Write header if the file is newly created
        writer.writerow(data)

# Example usage:
context_prompt = """
 You're an AI assistant to help students learn their course material via convertsations.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:
 {context_str}
 Instruction: Based on the above context, provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
 """



def get_response(prompt,message_history):
    # llm = OpenAI(model="gpt-3.5-turbo")
    try:
        prompt=f"{prompt} I-venture ISB"
        pplx_api_key=""
        groq_api=""
    # llm = Perplexity(
    # api_key=pplx_api_key, model="llama-2-70b-chat", temperature=0.2,api_base="https://api.perplexity.ai")
        llm = Groq(model="llama3-70b-8192", api_key=groq_api)
    #    llm = OpenAI(model="gpt-4-1106-preview")
        topk =3
    # service_context = ServiceContext.from_defaults(llm=llm)
        #query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
        chat_engine=CondensePlusContextChatEngine.from_defaults(retriever=hybrid_retriever,llm=llm,chat_history=message_history,context_prompt=context_prompt,condense_prompt=condense_prompt)
        nodes = hybrid_retriever.retrieve(prompt)
        context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        append_to_csv("context.csv",[prompt,context_str])
        response = chat_engine.chat(str(prompt))
        validating_prompt = ("""You are an intelligent bot designed to assist users on an organization's website by answering their queries. You'll be given a user's question and an associated answer. Your task is to determine if the provided answer effectively resolves the query. If the answer is unsatisfactory, return 0.\n
                            Query: {question}  
                            Answer: {answer}
                            Your Feedback:
                            """)
        feedback = llm.complete(validating_prompt.format(question=prompt,answer=response.response))
        if feedback.text==str(0):
            st.write("DISTANCE APPROACH")
            response , joined_text=answer_question(prompt)
            scores = rouge.get_scores(response, joined_text)
            message = {"role": "assistant", "content": response}
        # st.session_state.messages.append(message)
        # message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response)),)
            response_list = [response, prompt , scores]  
    #       df = pd.read_csv('logs/conversation_log.csv')
    #       new_row = {'Question': str(prompt), 'Answer': response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"]}
    #       df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    #       df.to_csv('logs/conversation_logs.csv', index=False)
    #       bucket = 'aiex' # already created on S3
    #       csv_buffer = StringIO()
    #       df.to_csv(csv_buffer)
    #       s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key= os.environ['ACCESS_KEY'])
    #       s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue())  
            return response_list                                 
        else:
            context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
        # st.write(context_str)
            scores=rouge.get_scores(response.response,context_str)
        # message = {"role": "assistant", "content": response.response}
        # st.session_state.messages.append(message)
        # message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
        # append_to_csv(prompt,context_str)
            response_list = [response.response , prompt , scores]
    #        df = pd.read_csv('logs/conversation_logs.csv')
    #        new_row = {'Question': str(prompt), 'Answer': response.response,'Unigram_Recall' : scores[0]["rouge-1"]["r"],'Unigram_Precision' : scores[0]["rouge-1"]["p"],'Bigram_Recall' : scores[0]["rouge-2"]["r"],'Bigram_Precision' : scores[0]["rouge-2"]["r"]}
    #        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    #        df.to_csv('logs/conversation_logs.csv', index=False)
    #        bucket = 'aiex' # already created on S3
    #        csv_buffer = StringIO()
    #        df.to_csv(csv_buffer)
    #        s3_resource= boto3.resource('s3',aws_access_key_id=os.environ['ACCESS_ID'],aws_secret_access_key=os.environ['ACCESS_KEY'])
    #        s3_resource.Object(bucket, 'conversation_log.csv').put(Body=csv_buffer.getvalue())
            return response_list 
    except:
        print("error")

def chat():
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [{"role": "assistant", "content": "Ask anything about I-Venture @ ISB!"}]
    if "message_history" not in st.session_state.keys():
        st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask anything about I-Venture @ ISB."),]
    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_history.append(ChatMessage(role=MessageRole.USER,content=str(prompt)))

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    prompt = st.session_state.messages[-1]["content"]
                    try:
                        response_content = get_response(prompt=f"{prompt} I-venture ISB",message_history=st.session_state.message_history)
                        st.write(response_content[0])

                    # st.write("hello")
                        # st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response_content[0])))
                # with st.chat_message("assistant"):
                        with st.spinner("Searching the web..."):
                        # if False:
                        # raise ValueError("This is an error message.")
                            groq_api=""
                            llm = OpenAI(model="gpt-4-1106-preview") 
                            #llm = Groq(model="llama3-70b-8192", api_key=groq_api)
                            #service_context = ServiceContext.from_defaults(llm=llm)
                                
                            prompt = st.session_state.messages[-1]["content"]
        #                    st.write("hello")
                            messages = [
                                {
                                        "role": "user",
                                    "content": prompt,
                                },
                                ]
        #                   st.write(messages)
        #                   st.session_state.messages.append({"role": "user", "content": prompt}) 
                            modified_messages = []
                            for message in messages:
                                if message["role"] == "user":
                                        modified_message = {"role": "user", "content": f"{message['content']} I-venture ISB"}
                                else:
                                        modified_message = message
                                modified_messages.append(modified_message)

                            # st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=prompt))

                            #    with st.chat_message("assistant"):
                            #       with st.spinner("Thinking..."):
                            client = openai.OpenAI(api_key="", base_url="https://api.perplexity.ai")
                            response = client.chat.completions.create(
                                        model="llama-3-sonar-large-32k-online",
                                        messages=modified_messages,
                                        )   
                            web_content = response.choices[0].message.content
                            query_engine=CondensePlusContextChatEngine.from_defaults(retriever=hybrid_retriever,llm=llm,chat_history=st.session_state.message_history,context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1,condense_prompt=condense_prompt)
                            enhanced_response = query_engine.chat(f"{prompt} web response:{web_content}")
        #                    st.write(web_content)
            #               enhanced_response = CondensePlusContextChatEngine.from_defaults(
            #                       query_engine,
        #                        context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1,
        #                         condense_prompt=condense_prompt,
        #                        chat_history=st.session_state.message_history
        #                    ).chat(f"{prompt} {web_content}")
        #                        enhanced_response_content = enhanced_response.response if hasattr(enhanced_response, 'response') else str(enhanced_response)

                                # st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=user_input))
                            st.session_state.messages.append({"role": "assistant", "content": f"{response_content[0]}\n\n Enhanced Answer:\n {enhanced_response.response}"})
                            st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"{response_content[0]}\n {enhanced_response.response}"))
        #
                            st.write("Enhanced Answer: ")
                            st.write(enhanced_response.response)
                    except:
                        try:
                            with st.spinner("Searching the web..."):
                            # if False:
                            # raise ValueError("This is an error message.")
                                groq_api=""
                                llm = OpenAI(model="gpt-4-1106-preview") 
                                #llm = Groq(model="llama3-70b-8192", api_key=groq_api)
                                #service_context = ServiceContext.from_defaults(llm=llm)
                                    
                                prompt = st.session_state.messages[-1]["content"]
            #                    st.write("hello")
                                messages = [
                                    {
                                            "role": "user",
                                        "content": prompt,
                                    },
                                    ]
            #                   st.write(messages)
            #                   st.session_state.messages.append({"role": "user", "content": prompt}) 
                                modified_messages = []
                                for message in messages:
                                    if message["role"] == "user":
                                            modified_message = {"role": "user", "content": f"{message['content']} I-venture ISB"}
                                    else:
                                            modified_message = message
                                    modified_messages.append(modified_message)

                                # st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=prompt))

                                #    with st.chat_message("assistant"):
                                #       with st.spinner("Thinking..."):
                                client = openai.OpenAI(api_key="", base_url="https://api.perplexity.ai")
                                response = client.chat.completions.create(
                                            model="llama-3-sonar-large-32k-online",
                                            messages=modified_messages,
                                            )   
                                web_content = response.choices[0].message.content
                                query_engine=CondensePlusContextChatEngine.from_defaults(retriever=hybrid_retriever,llm=llm,chat_history=st.session_state.message_history,context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1,condense_prompt=condense_prompt)
                                enhanced_response = query_engine.chat(f"{prompt} web response:{web_content}")
            #                    st.write(web_content)
                #               enhanced_response = CondensePlusContextChatEngine.from_defaults(
                #                       query_engine,
            #                        context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1,
            #                         condense_prompt=condense_prompt,
            #                        chat_history=st.session_state.message_history
            #                    ).chat(f"{prompt} {web_content}")
            #                        enhanced_response_content = enhanced_response.response if hasattr(enhanced_response, 'response') else str(enhanced_response)

                                    # st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=user_input))
                                st.session_state.messages.append({"role": "assistant", "content": f"{response_content[0]}\n\n Enhanced Answer:\n {enhanced_response.response}"})
                                st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=enhanced_response.response))
            #
                                st.write("Enhanced Answer: ")
                                st.write(enhanced_response.response)
                        except:
                            with st.spinner("Searching the web..."):
                                messages = [
                                    {
                                            "role": "user",
                                        "content": prompt,
                                    },
                                    ]
            #                   st.write(messages)
            #                   st.session_state.messages.append({"role": "user", "content": prompt}) 
                                modified_messages = []
                                for message in messages:
                                    if message["role"] == "user":
                                            modified_message = {"role": "user", "content": f"{message['content']} I-venture ISB"}
                                    else:
                                            modified_message = message
                                    modified_messages.append(modified_message)
                                # st.session_state.message_history.append(ChatMessage(role=MessageRole.USER, content=prompt))

                                #    with st.chat_message("assistant"):
                                #       with st.spinner("Thinking..."):
                                client = openai.OpenAI(api_key="", base_url="https://api.perplexity.ai")
                                response = client.chat.completions.create(
                                            model="llama-3-sonar-large-32k-online",
                                            messages=modified_messages,
                                            )   
                                web_content = response.choices[0].message.content
                                st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=str(web_content)))
                                st.write(str(web_content))

def main():
    st.title("I-Venture @ ISB Chatbot")
    chat()

if __name__ == "__main__":
    main()
