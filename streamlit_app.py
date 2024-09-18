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
from langchain.document_loaders import PyPDFLoader
import requests
from bs4 import BeautifulSoup

#Stremlit App title
st.title("Website Intelligence")

#LLM
api_key = "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"

llm = ChatGroq(groq_api_key = api_key, model_name = 'llama-3.1-70b-versatile', temperature = 0.2, top_p = 0.2)

#Embedding
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#User inputs for sitemap URLs and filter keywords
with st.form(key = 'document_form'):
  sitemap_input = st.text_area("Enter Sitemap URLs (https://www.example.com/sitemap.xml):")
  filter_input = st.text_area("Enter keywords to filter (comma-seperated):")

  submit_button = st.form_submit_button(label = "Load Documents")

#Processing user inputs
sitemap_urls = [url.strip() for url in sitemap_input.split(",") if url.strip()]
filter_urls = [keyword.strip() for keyword in filter_input.split(",") if keyword.strip()]


st.cache_data(show_spinner = False)

def fetch_documents(sitemap_urls, filter_urls):
  loaded_docs = []
  for sitemap_url in sitemap_urls:
    try:
      response = requests.get(sitemap_url)
      sitemap_content = response.content

      #Parse sitemap URL
      soup = BeautifulSoup(sitemap_content, 'xml')
      urls = [loc.text for loc in soup.find_all('loc')]

      #Filter URLs
      selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]

  #Append URLs to the main list
  #filtered_urls.extend(selected_urls)

      for url in selected_urls:
        try:
          #st.write(f"Loading URL: {url}")
          loader = WebBaseLoader(url)
          docs = loader.load()
          for doc in docs:
            doc.metadata["source"] = url
          loaded_docs.extend(docs)
          #st.success("Successfully loaded document")
        except Exception as e:
          st.error("Error")
    except Exception as e:
        st.error("Error")
  return loaded_docs

#Load Documents

if submit_button:
  try:
    loaded_docs = fetch_documents(sitemap_urls, filter_urls)
    st.write(f"Loaded documents: {len(loaded_docs)}")
  
    #Craft ChatPrompt Template
    prompt = ChatPromptTemplate.from_template(
      """
      You are a Life Insurance specialist who needs to answer queries based on the information provided in the websites only. Please follow all the websites, and answer as per the same.
      Do not answer anything except from the website information which has been entered.
      Do not skip any information from the context. Answer appropriately as per the query asked.
      Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the websites, if asked.
      Generate tabular data wherever required to classify the difference between different parameters of policies.
      I will tip you with a $1000 if the answer provided is helpful.
      <context>
      {context}
      </context>
      Question: {input}""")
  
      #Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1000,
          chunk_overlap  = 100,
          length_function = len,
    )
      
    document_chunks = text_splitter.split_documents(loaded_docs)
  
      
      
      #Vector database storage
    vector_db = FAISS.from_documents(document_chunks, hf_embedding)
      
      
      #query = "What are the premium payment modes available for TATA life insurance plan & HDFC Life insurance plan?"
      #docs = vector_db.similarity_search(query)
      #for doc in docs:
        #print({doc.metadata['source']})
        #print({doc.page_content})
      
      #Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)
      
      #Retriever from Vector store
    retriever = vector_db.as_retriever()
      
      #Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

  except Exception as e:
    st.error("Error")
  
      
user_query = st.text_input("Ask a question:")
if st.button("Get Answer"):
  try:
    response = retrieval_chain.invoke({"input": user_query})
      
    st.write("Answer")
    st.write(response['answer'])
      
    st.write("Sources:")
    for doc in response.get('source_documents', []):
      st.write(f" {doc.metadata['source']}")
  except Exception as e:
    st.error("Error")
