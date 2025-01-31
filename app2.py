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
# With this:
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # or your Bedrock region
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    try:
        current_dir = os.getcwd()
        data_dir = os.path.join(current_dir, "data")
        st.sidebar.write(f"Current working directory: {current_dir}")
        st.sidebar.write(f"Data directory path: {data_dir}")
        
        if not os.path.exists(data_dir):
            st.error(f"Data directory does not exist at {data_dir}")
            return None
        
        st.sidebar.write(f"Files in data directory: {os.listdir(data_dir)}")
        
        loader = PyPDFDirectoryLoader(data_dir)
        documents = loader.load()
        
        if not documents:
            st.error("No documents loaded. Check if PDFs are valid and accessible.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        docs = text_splitter.split_documents(documents)

        if not docs:
            st.error("Document splitting resulted in empty chunks.")
            return None

        # Verify document content
        valid_docs = []
        for doc in docs:
            if doc.page_content.strip():  # Check for non-empty content
                valid_docs.append(doc)
        
        if not valid_docs:
            st.error("No valid text content found in documents.")
            return None

        return valid_docs

    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

## Vector embeddings and vector store
def get_vector_store(docs):
    if not docs:
        raise ValueError("No documents provided for vector store creation")
    
    try:
        vectorstore_faiss = FAISS.from_documents(
            docs,
            bedrock_embeddings
        )

        vectorstore_faiss.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False


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
Human: Use the following pieves of context to provide a good detailed and relevant answer to the question at the end. If you don't know the answer, be honest and say you don't know. Don't make up the answers
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
                if docs:
                    success = get_vector_store(docs)
                    if success:
                        st.success("Vector store updated successfully!")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("No documents to process")
    
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings, allow_dangerous_deserialization=True  # Explicitly enable
            )
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")



if __name__ == "__main__":
    main()
