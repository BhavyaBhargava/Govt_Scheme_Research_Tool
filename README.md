# Scheme Research Application

## Overview

This project is a **Scheme Research Tool** designed to simplify the extraction and presentation of information about government schemes. The tool processes scheme-related articles and provides users with concise summaries based on four essential criteria: **Scheme Benefits**, **Scheme Application Process**, **Eligibility**, and **Required Documents**. Additionally, it supports interactive querying of the article's content, offering users both answers and the source of information.

## Features

- **Input Options**:  
  - Load URLs or upload text files containing URLs to fetch article content.

- **Automated Content Processing**:  
  - Utilize LangChain's UnstructuredURL Loader to parse and process content efficiently.  
  - Split and structure text for enhanced usability.

- **Advanced Search and Retrieval**:  
  - Generate embedding vectors using OpenAI's embeddings.  
  - Implement similarity search using the FAISS library for rapid and accurate information retrieval.  
  - Store and reuse FAISS indexes locally as a pickle file.

- **Interactive Query System**:  
  - Input user questions and receive precise answers based on the article's content.  
  - Provide the source URL and a brief summary along with the answer.

- **Streamlined User Interface**:  
  - Web application built with Streamlit for ease of use and accessibility.  
  - Sidebar input for URL submission.  
  - Intuitive buttons for processing and interacting with content.

- **Persistent Data Storage**:  
  - Save embeddings and FAISS indexes locally for quick future access.

## Technical Details

- **Main Application**:  
  - `main.py`: Streamlit application script managing the interface and backend processes.

- **Dependencies**:  
  - `requirements.txt`: Contains the list of Python libraries required for the project.

- **Configuration**:  
  - `.config`: Store OpenAI API keys securely.

- **FAISS Index File**:  
  - `faiss_store_openai.pkl`: Save and load indexed embeddings.

## Solution Workflow

1. Users input URLs or upload text files in the sidebar.  
2. The system fetches and processes content from the URLs.  
3. Text is split, embedding vectors are generated, and a FAISS index is built.  
4. Users can interact by asking questions, and the system retrieves answers based on indexed content.

## Usage

- Install the required dependencies from `requirements.txt`.  
- Run `main.py` to launch the Streamlit web application.  
- Submit scheme URLs and process them for structured summaries and interactive queries.  

---

This tool bridges the gap between raw information and user understanding, ensuring efficient access to government scheme details.
