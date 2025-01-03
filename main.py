import configparser
import streamlit as st
import pickle
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load configuration from .config file
config = configparser.ConfigParser()
config.read("config.config")

# Get API key from config file
GROQ_API_KEY = config.get("api_keys", "groq_api_key", fallback=None)

# Verify API key
def verify_api_key():
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please set it in the `config.config` file under [api_keys].")
        st.stop()

# Load URLs from a text file
def load_urls_from_file(file):
    content = file.getvalue().decode()
    urls = [url.strip() for url in content.split('\n') if url.strip()]
    return urls

# Process URLs using UnstructuredURLLoader
def process_urls(urls):
    try:
        with st.spinner("Loading content from URLs..."):
            loader = UnstructuredURLLoader(urls=urls)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            return texts
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")
        return None

# Create vector store for question-answering
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    with open("faiss_store_Meta.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

# Generate summaries for specific sections using ChatGroq and vector store
def generate_scheme_summary(vector_store):
    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
    
    summary_aspects = {
        "benefits": "What are the main benefits of this scheme?",
        "process": "What is the application process for this scheme?",
        "eligibility": "What are the eligibility criteria for this scheme?",
        "documents": "What documents are required for this scheme?"
    }
    
    summaries = {}
    for aspect, question in summary_aspects.items():
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        response = retrieval_chain.invoke(question)
        summaries[aspect] = response["result"]
        
    return summaries

# Main application
def main():
    verify_api_key()

    # Initialize session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar
    with st.sidebar:
        st.header("Input URLs")
        urls_input = st.text_area("Enter URLs (one per line)")
        uploaded_file = st.file_uploader("Or upload a text file with URLs", type=['txt'])
        process_button = st.button("Process URLs")

    # Main content
    if not st.session_state["summary_generated"]:
        st.title("Scheme Research Tool")
        st.write("This tool extracts content from URLs about government schemes, summarizes it into four sections, and allows you to ask questions.")

    if process_button:
        urls = []
        if urls_input:
            urls.extend([url.strip() for url in urls_input.split('\n') if url.strip()])
        if uploaded_file:
            urls.extend(load_urls_from_file(uploaded_file))
        
        if urls:
            texts = process_urls(urls)
            if texts:
                with st.spinner("Creating vector store..."):
                    vector_store = create_vector_store(texts)
                    st.session_state.vector_store = vector_store
                
                with st.spinner("Generating summaries..."):
                    summaries = generate_scheme_summary(vector_store)
                    
                    st.session_state["summary_generated"] = True
                    
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
    if st.session_state["summary_generated"]:
        st.title("Ask Questions")
        
        new_question = st.text_input("Enter your question about the schemes")
        
        if new_question:
            try:
                vector_store = st.session_state.vector_store
                
                with st.spinner("Finding answer..."):
                    llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True
                    )
                    
                    response = qa_chain.invoke(new_question)
                    st.session_state.qa_history.append({
                        "question": new_question,
                        "answer": response["result"]
                    })
            
            except (FileNotFoundError, AttributeError):
                st.error("Please process URLs first!")
        
        # Display Q&A history
        for qa in st.session_state.qa_history:
            st.subheader("Q: " + qa["question"])
            st.write("A: " + qa["answer"])
            st.divider()

if __name__ == "__main__":
    main()