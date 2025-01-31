import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model for generate embeddings.

from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

#from langchain_community.embeddings import BedrockEmbeddings
#from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store

from langchain_community.vectorstores import FAISS
## LLM Models
from langchain.prompts import PromptTemplate
#from langchain.chains import retrieval_qa
from langchain.chains import RetrievalQA


## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    data_folder_path = os.path.abspath("data")  # This gives the full path of 'data' folder
    loader = PyPDFDirectoryLoader(data_folder_path)
    documents = loader.load()
    
    # In our testing Character split works better with this PDF data
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

     # Check if splitting worked correctly
    if not docs:
        print("No documents after splitting.")
    else:
        print(f"Split into {len(docs)} chunks.")

    return docs

## Vector embeddings and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Antropic Model
    llm=Bedrock(model_id="anthropic.claude-v2", client=bedrock,
                model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    ##Create the llama2 model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':512})
    return llm

Prompt_template = """
Human: Use the following pieves of context to provide a concise anwer to the question at the end. If you don't know the answer, be honest and say you don't know. Don't make up the answers
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=Prompt_template, input_variables=["context","questions"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")



if __name__ == "__main__":
    main()
