# RAG Application

A Retrieval-Augmented Generation (RAG) application that enables intelligent question-answering on user document using LangChain, HuggingFace models, and ChromaDB vector database.

## Overview

This project implements a complete RAG pipeline that processes the PDF document, creates searchable embeddings, and uses a Large Language Model to answer user questions based on the retrieved context. The system combines document retrieval with generative AI to provide accurate, context-based responses.

## System Architecture

![RAG System Architecture](./system_architecture.png)

The system follows a two-phase architecture:

1. **Indexing Pipeline**: PDF → Chunks → Embeddings → Vector Database
2. **Query Pipeline**: Question → Retrieve → Context + Prompt → LLM → Answer

## Technologies Used

| Component | Technology |
|-----------|------------|
| Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | TinyLlama-1.1B-Chat |
| Vector Database | ChromaDB |

## Project Structure

```bash
RAG_APPLICATION/
│
├── data/
│   ├── documents/
│   │   └── User_source.pdf    # Source document
│   └── chroma_db/                       # Persisted vector store
│
├── config.py              # Configuration settings and paths
├── indexing.py            # Document processing and embedding pipeline
├── retrieval.py           # RAG chain creation and query handling
├── main.py                # Entry point and interactive interface
├── utils.py               # Utility functions
├── test.py                # Testing utilities
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Configuration

#### Model Settings
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```


#### Chunking Parameters
```python
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
```

#### Retrieval Settings
```python
TOP_K = 2                           # Number of documents to retrieve
COLLECTION_NAME = "nepal_constitution"
```

## Usage

### Running the Application
```bash
python main.py
```


