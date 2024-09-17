import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup
import requests
import faiss
import numpy as np

#LLM
api_key = "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"

llm = ChatGroq(groq_api_key = api_key, model_name = 'llama3-8b-8192', temperature = 0.2, top_p = 0.2)

# Define a prompt template for LangChain
prompt_template = """Given the following information extracted from websites:

{website_data}

Can you summarize and compare the key details, especially the differences in the insurance policies offered by various companies?"""

prompt = PromptTemplate(template=prompt_template, input_variables=["website_data"])
comparison_chain = LLMChain(llm=llm, prompt=prompt)

# Function to scrape website content (text extraction)
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract text from website's relevant sections
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    
    return text

# Function to embed the website content using FAISS
def embed_website_content(text):
    vectors = np.array([llm.embed(text_chunk) for text_chunk in text.split("\n")])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    
    return index, vectors

# Function to compare multiple websites
def compare_websites(urls):
    website_data = []
    
    for url in urls:
        content = scrape_website(url)
        _, _ = embed_website_content(content)
        website_data.append(content)
    
    combined_data = "\n\n".join(website_data)
    comparison_result = comparison_chain.run(website_data=combined_data)
    
    return comparison_result

# Streamlit UI
st.title("Website Intelligence and Comparison Chatbot")

st.write("""
This chatbot scrapes information from insurance websites and compares the policies.
Enter the URLs of the websites to compare.
""")

# Text input for URLs
url_input = st.text_area("Enter website URLs (comma-separated)", value="https://www.bajajallianz.com/insurance-policy-page, https://www.hdfclife.com/insurance-policy-page")

if st.button("Compare Websites"):
    urls = [url.strip() for url in url_input.split(",")]
    
    # Display loading spinner while processing
    with st.spinner('Comparing websites...'):
        comparison_result = compare_websites(urls)
    
    # Display the result
    st.subheader("Comparison Result")
    st.write(comparison_result)
