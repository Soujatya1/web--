import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

#Steamlite title
st.title("Website Intelligence and Comparer")

#LLM
api_key = "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"

llm = ChatGroq(groq_api_key = api_key, model_name = 'llama-3.1-70b-versatile', temperature = 0, top_p = 0.2)

#Embedding
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

urls = ["https://www.hdfclife.com/term-insurance-plans",
"https://www.hdfclife.com/term-insurance-plans/click-2-protect-super",
"https://www.hdfclife.com/term-insurance-plans/sanchay-legacy",
"https://www.hdfclife.com/term-insurance-plans/click-2-protect-elite",
"https://www.hdfclife.com/term-insurance-plans/term-with-return-of-premium-plan",
"https://www.hdfclife.com/term-insurance-plans/quick-protect",
"https://www.hdfclife.com/term-insurance-plans/saral-jeevan-bima"]

loaded_docs = []

def load_document(url):
  try:
    st.spinner("Loading URL...")
    loader = WebBaseLoader(url)
    return loader.load(), url
  except Exception as e:
    st.error("Error")
    return [], url

#Parallel doc loading

def load_all_documents(urls):
  loaded_docs = []
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(load_document, url) for url in urls]
    for future in as_completed(futures):
      try:
        docs, url = future.result()
        loaded_docs.extend(docs)
        if docs:
          st.write("Loaded successfully")
      except Exception as e:
        st.error("Error loading")
  return loaded_docs


st.write(f"Loaded urls: {len(urls)}")

#Craft ChatPrompt Template
prompt = ChatPromptTemplate.from_template(
"""
You are an HDFC Life Insurance specialist who needs to answer queries based on the information provided in the websites. Please follow all the websites, and answer as per the same.
Do not answer anything out of the website information.
Do not skip any information from the context. Answer appropriately as per the query asked.
Now, being an excellent HDFC Life Insurance agent, you need to compare your policies against the other company's policies in the websites.
Generate tabular data wherever required to classify the difference between different parameters of policies.
I will tip you with a $1000 if the answer provided is helpful.
<context>
{context}
</context>
Question: {input}""")

#Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap  = 200,
    length_function = len,
)

document_chunks = text_splitter.split_documents(loaded_docs)

#Vector database storage
vector_db = FAISS.from_documents(document_chunks, hf_embedding)

#Streamlit user query
query = st.text_input("Enter your question")

if query:
    #Perform similarity search
    docs = vector_db.similarity_search(query)

    #Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    #Retriever from Vector store
    retriever = vector_db.as_retriever()

    #Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({"input": query})

    st.write(response['answer'])
