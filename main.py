"""
Main entry point for the RAG application.
Nepal Constitution Q&A System
"""

import os

import config
import utils
from indexing import run_indexing_pipeline, get_embeddings, load_vector_store
from retrieval import run_retrieval_pipeline, query


def main():
    # Check if vector store already exists
    chroma_exists = os.path.exists(config.CHROMA_DB_DIR) and os.listdir(config.CHROMA_DB_DIR)
    
    if not chroma_exists:
        # Run indexing pipeline (first time only)
        utils.print_separator("FIRST RUN - INDEXING DOCUMENTS")
        run_indexing_pipeline()
    else:
        print("Vector store already exists. Skipping indexing.")
    
    # Initialize retrieval pipeline
    rag_chain, vector_store = run_retrieval_pipeline()
    
    # Sample queries to test the system
    utils.print_separator("TESTING RAG SYSTEM")
    
    test_questions = [
        "What are the fundamental rights mentioned in the constitution?",
        "What is the structure of the federal government?",
        "What does the constitution say about citizenship?"
    ]
    
    for question in test_questions:
        utils.print_separator()
        query(rag_chain, question)
    
    # Interactive mode
    utils.print_separator("INTERACTIVE MODE")
    print("Enter your questions about Nepal's Constitution (type 'quit' to exit)")
    
    while True:
        user_question = input("\nYour question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if user_question:
            query(rag_chain, user_question)


if __name__ == "__main__":
    main()