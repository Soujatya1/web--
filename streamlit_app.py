import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import pickle
import os
from joblib import Parallel, delayed

# Initialize session state variables to persist data across reruns
if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None

# Streamlit UI
st.title("Knowledge Management Chatbot")

api_key = "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"

sitemap_urls_input = st.text_area("Enter sitemap URLs (one per line):")
filter_words_input = st.text_area("Enter filter words (one per line):")

if st.button("Load and Process"):
    sitemap_urls = sitemap_urls_input.splitlines()
    filter_urls = filter_words_input.splitlines()
    
    all_urls = []
    filtered_urls = []
    st.session_state['loaded_docs'] = []
    
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url)
            sitemap_content = response.content

            # Parse sitemap URL
            soup = BeautifulSoup(sitemap_content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]

            # Filter URLs
            selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]

            # Append URLs to the main list
            filtered_urls.extend(selected_urls)

            for url in filtered_urls:
                try:
                    #st.write(f"Loading URL: {url}")
                    loader = WebBaseLoader(url)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = url

                    st.session_state['loaded_docs'].extend(docs)
                    #st.write("Successfully loaded document")
                except Exception as e:
                    st.write(f"Error loading {url}: {e}")

        except Exception as e:
            st.write(f"Error processing sitemap {sitemap_url}: {e}")
    
    st.write(f"Loaded documents: {len(st.session_state['loaded_docs'])}")
    
    # LLM and Embeddings Initialization
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)
        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Craft ChatPrompt Template
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Life Insurance specialist who needs to answer queries based on the information provided in the websites only. Please follow all the websites, and answer as per the same.

            Do not answer anything except from the website information which has been entered. Please do not skip any information from the tabular data in the website.

            Do not skip any information from the context. Answer appropriately as per the query asked.

            Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the websites, if asked.

            Generate tabular data wherever required to classify the difference between different parameters of policies.

            <context>
            {context}
            </context>

            Question: {input}"""
        )
        
        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )

        documents = st.session_state['loaded_docs']
        document_chunks = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc.page_content)
            for split_text in split_texts:
                document_chunks.append({"page_content": split_text, "metadata": doc.metadata})

        #document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
        st.write(f"Number of chunks: {len(document_chunks)}")

        # Embedding and Vector Database Storage
        batch_size = 16 # Adjust batch size according to your memory limits
        vectors = []

        # Check if cached embeddings exist
        embeddings_cache_file = "cached_embeddings.pkl"
        if os.path.exists(embeddings_cache_file):
            with open(embeddings_cache_file, 'rb') as f:
                vectors = pickle.load(f)
        else:
            # Parallel embedding with batching
            def embed_chunk(chunk):
                return hf_embedding.embed_documents([c['page_content'] for c in chunk])
            
            vectors = Parallel(n_jobs=-1)(delayed(embed_chunk)(document_chunks[i:i + batch_size]) 
                                          for i in range(0, len(document_chunks), batch_size))
            # Flatten the list of vectors
            vectors = [item for sublist in vectors for item in sublist]

            # Cache the embeddings
            with open(embeddings_cache_file, 'wb') as f:
                pickle.dump(vectors, f)

        # Initialize FAISS index with IVF
        import faiss
        dimension = len(vectors[0]) # Dimensionality of embeddings
        nlist = 100 # Number of clusters for IVF

        quantizer = faiss.IndexFlatL2(dimension)
        ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train and add vectors to FAISS
        ivf_index.train(vectors)
        ivf_index.add(vectors)

        # Store in FAISS vector database
        st.session_state['vector_db'] = FAISS(embeddings=hf_embedding, index=ivf_index)

        # Stuff Document Chain Creation
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Retriever from Vector store
        retriever = st.session_state['vector_db'].as_retriever()

        # Create a retrieval chain
        st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)

# Query Section
query = st.text_input("Enter your query:")
if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        response = st.session_state['retrieval_chain'].invoke({"input": query})
        st.write("Response:")
        st.write(response['answer'])
    else:
        st.write("Please load and process documents first.")
