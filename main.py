import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Scheme Research Tool", layout="wide")

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'summary_generated' not in st.session_state:
    st.session_state.summary_generated = False

# Verify API key
def verify_api_key():
    if not os.getenv("GROQ_API_KEY"):
        st.error("Groq API key not found. Please set GROQ_API_KEY in .env file.")
        st.stop()

# Load URLs from file
def load_urls_from_file(file):
    content = file.getvalue().decode()
    urls = [url.strip() for url in content.split('\n') if url.strip()]
    return urls

# Process URLs
def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(data)
    return texts

# Create vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    with open("faiss_store_Meta.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

# Generate scheme summary using ChatGroq
def generate_scheme_summary(vector_store):
    system_message = "You are an intelligent assistant summarizing government schemes."
    human_message = "{text}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])

    llm = ChatGroq(model="llama3-70b-8192")
    chain = prompt | llm

    summary_aspects = {
        "benefits": "What are the main benefits of these schemes?",
        "process": "What is the application process for these schemes?",
        "eligibility": "Who is eligible for these schemes?",
        "documents": "What documents are required for these schemes?"
    }
    
    summaries = {}
    for aspect, question in summary_aspects.items():
        response = chain.invoke({"text": question})
        summaries[aspect] = response.content

    return summaries

# Main application
def main():
    verify_api_key()

    # Sidebar
    with st.sidebar:
        st.header("Input URLs")
        urls_input = st.text_area("Enter URLs (one per line)")
        uploaded_file = st.file_uploader("Or upload a text file with URLs", type=['txt'])
        process_button = st.button("Process URLs")

    # Main content
    if not st.session_state.summary_generated:
        st.title("Welcome to the Scheme Research Tool")
        st.write("This tool helps you understand government schemes better. Please enter URLs in the sidebar or upload a file to get started.")

    if process_button:
        urls = []
        if urls_input:
            urls.extend([url.strip() for url in urls_input.split('\n') if url.strip()])
        if uploaded_file:
            urls.extend(load_urls_from_file(uploaded_file))
        
        if urls:
            with st.spinner("Processing URLs..."):
                texts = process_urls(urls)
                vector_store = create_vector_store(texts)
                summaries = generate_scheme_summary(vector_store)
                
                st.session_state.summary_generated = True
                st.empty()
                
                st.title("Scheme Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Benefits")
                    st.write(summaries["benefits"])
                    
                    st.subheader("Application Process")
                    st.write(summaries["process"])
                
                with col2:
                    st.subheader("Eligibility")
                    st.write(summaries["eligibility"])
                    
                    st.subheader("Required Documents")
                    st.write(summaries["documents"])

    # Question-answering section
    if st.session_state.summary_generated:
        st.title("Ask Questions")
        
        new_question = st.text_input("Enter your question about the schemes")
        
        if new_question:
            try:
                with open("faiss_store_meta.pkl", "rb") as f:
                    vector_store = pickle.load(f)
                
                with st.spinner("Finding answer..."):
                    system_message = "You are an intelligent assistant answering questions."
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        ("human", "{text}")
                    ])
                    llm = ChatGroq(model="llama3-8b-8192")
                    chain = prompt | llm
                    response = chain.invoke({"text": new_question})

                    st.session_state.qa_history.append({"question": new_question, "answer": response.content})
            
            except FileNotFoundError:
                st.error("Please process URLs first!")
        
        # Display Q&A history
        for qa in st.session_state.qa_history:
            st.subheader("Q: " + qa["question"])
            st.write("A: " + qa["answer"])
            st.divider()

if __name__ == "__main__":
    main()
