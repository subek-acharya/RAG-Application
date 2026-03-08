"""
INDEXING PIPELINE: Loda PDF → Chunks → Embeddings → Vector Store
Run once to prepare your knowledge base.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config
import utils


def load_pdf(pdf_path):
    """Load PDF document and return list of pages."""
    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks."""
    print(f"Splitting documents (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    return splits


def get_embeddings():
    """Initialize HuggingFace embeddings model."""
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return embeddings


def create_vector_store(splits, embeddings):
    """Create and persist Chroma vector store."""
    utils.ensure_directory_exists(config.CHROMA_DB_DIR)
    
    print(f"Creating vector store at: {config.CHROMA_DB_DIR}")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DB_DIR
    )
    
    print(f"Vector store created with {len(splits)} documents")
    return vector_store


def load_vector_store(embeddings):
    """Load existing Chroma vector store."""
    print(f"Loading existing vector store from: {config.CHROMA_DB_DIR}")
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DB_DIR
    )
    return vector_store


def run_indexing_pipeline():
    """Run the complete indexing pipeline."""
    utils.print_separator("INDEXING PIPELINE")
    
    # Load PDF
    documents = load_pdf(config.PDF_PATH)
    
    # Split documents
    splits = split_documents(documents)
    
    # Get embeddings
    embeddings = get_embeddings()
    
    # Create vector store
    vector_store = create_vector_store(splits, embeddings)
    
    utils.print_separator("INDEXING COMPLETE")
    return vector_store


if __name__ == "__main__":
    run_indexing_pipeline()