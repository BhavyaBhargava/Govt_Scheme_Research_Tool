# Scheme Research Tool

## Overview
This project is a **Scheme Research Tool** designed to automate the analysis of government scheme information. The tool processes scheme-related articles and provides structured summaries covering four essential aspects: **Scheme Benefits**, **Scheme Application Process**, **Eligibility**, and **Required Documents**. It features an interactive querying system that allows users to ask specific questions about the schemes.

## Key Features

### Content Processing
* Direct URL input or bulk upload via text files
* Robust content extraction using LangChain's UnstructuredURL Loader
* Efficient text splitting with RecursiveCharacterTextSplitter
* Error handling for failed URL processing

### Advanced Search and Retrieval
* High-performance embedding generation using Sentence Transformers (all-mpnet-base-v2)
* Efficient similarity search implementation using FAISS
* Local storage of FAISS indexes for quick access
* Retrieval Augmented Generation (RAG) for accurate responses

### Language Model Integration
* Powered by Groq's LLaMA 3 (8B parameters)
* Low-latency response generation
* Context-aware question answering
* Robust summarization capabilities

### User Interface
* Clean, intuitive web interface built with Streamlit
* Real-time processing status updates
* Organized presentation of scheme summaries
* Interactive Q&A section with history tracking

## Technical Architecture

### Core Components
1. **Document Processing**
   * UnstructuredURLLoader for content extraction
   * RecursiveCharacterTextSplitter for text chunking
   * Optimal chunk size: 1000 with 200 overlap

2. **Embedding System**
   * Model: sentence-transformers/all-mpnet-base-v2
   * Local processing without API dependencies
   * Cost-effective and reliable performance

3. **Vector Storage**
   * FAISS implementation for similarity search
   * Persistent storage through pickle serialization
   * Efficient retrieval mechanisms

### Dependencies
```
streamlit
langchain
langchain-community
faiss-cpu
unstructured
groq
sentence-transformers
langchain_groq
langchain_huggingface
```

## Installation and Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd scheme-research-tool
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
   * Create a `config.config` file in the root directory
   * Add your Groq API key:
```ini
[api_keys]
groq_api_key = your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Using the tool:
   * Enter URLs directly in the sidebar text area or upload a text file containing URLs
   * Click "Process URLs" to initiate content analysis
   * View generated summaries in four structured sections
   * Use the Q&A section to ask specific questions about the schemes

## File Structure

```
scheme-research-tool/
├── main.py              # Main Streamlit application
├── requirements.txt     # Project dependencies
├── config.config       # API key configuration
└── faiss_store_Meta.pkl # Stored FAISS index
```

## Technical Advantages

### Embedding Choice
* Uses HuggingFace Sentence Transformers instead of OpenAI embeddings
* Eliminates API costs and rate limits
* Supports offline operation
* Complete control over embedding process
* Comparable performance to commercial alternatives

### Vector Search
* FAISS implementation for efficient similarity search
* Scalable for large document collections
* Optimized memory usage
* Fast retrieval times

## Error Handling

* Robust URL processing error management
* Clear user feedback for processing status
* Graceful handling of API failures
* Session state management for consistent user experience

## Contributing
Feel free to submit issues and enhancement requests.

## License
This repository and its contents are strictly protected under copyright law. Unauthorized copying, distribution, modification, or use of any code, files, or materials contained within this repository is expressly prohibited without prior written consent from the owner.

####Key Terms and Conditions:
No License Granted:
Unless explicitly stated, this repository is not licensed for public or private use. Any usage without the owner’s written permission constitutes copyright infringement.

####Prohibited Actions:

Forking or cloning this repository without authorization.
Copying code, solutions, or files for personal or commercial use.
Reusing or redistributing the content, in whole or part, without explicit permission.
####Legal Consequences:
Violators may face legal actions under applicable intellectual property laws, including but not limited to claims for damages, injunctions, and attorney fees.

###Contact for Permissions:
For inquiries or permission requests, please contact me directly via the email associated with this repository.

---

For more information or support, please [create an issue](link-to-issues) in this repository.
