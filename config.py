"""
Configuration parameters for the RAG application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now use the API key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

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
# LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# LLM_MODEL = "microsoft/phi-2"
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
USE_4BIT = True  # Enable 4-bit quantization

# Rerank settings
USE_RERANK = True  # Set to False to disable reranking
RERANK_MODEL = "rerank-v3.5"
INITIAL_TOP_K = 15  # Fetch more documents initially for reranking
FINAL_TOP_K = 5     # Keep top documents after reranking

# Retrieval parameters (used when reranking is disabled)
TOP_K = 4  # Number of documents to retrieve

# Collection name for Chroma
COLLECTION_NAME = "nepal_constitution"