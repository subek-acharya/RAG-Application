"""
Configuration parameters for the RAG application.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Document settings
PDF_PATH = os.path.join(DOCUMENTS_DIR, "Constitution_of_Nepal.pdf")

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (choose based on your GPU memory)
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Retrieval parameters
TOP_K = 4  # Number of documents to retrieve

# Collection name for Chroma
COLLECTION_NAME = "nepal_constitution"