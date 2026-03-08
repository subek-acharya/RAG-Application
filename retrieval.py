"""
Retrieval and generation pipeline.
"""

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

import config
import utils
from indexing import get_embeddings, load_vector_store


def get_llm():
    """Initialize HuggingFace LLM."""
    print(f"Loading LLM: {config.LLM_MODEL}")
    
    # Create transformers pipeline
    pipe = pipeline(
        "text-generation",
        model=config.LLM_MODEL,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def get_retriever(vector_store):
    """Create retriever from vector store."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K}
    )
    return retriever


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm):
    """Create the RAG chain."""
    
    # Define prompt template
    template = """Use the following context to answer the question. 
    If you don't know the answer based on the context, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate.from_template(template)
    
    # Create chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def query(rag_chain, question):
    """Query the RAG chain and return response."""
    print(f"\nQuestion: {question}")
    response = rag_chain.invoke(question)
    print(f"Answer: {response}")
    return response


def run_retrieval_pipeline():
    """Initialize retrieval components."""
    utils.print_separator("INITIALIZING RETRIEVAL PIPELINE")
    
    # Load embeddings and vector store
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    
    # Create retriever
    retriever = get_retriever(vector_store)
    
    # Load LLM
    llm = get_llm()
    
    # Create RAG chain
    rag_chain = create_rag_chain(retriever, llm)
    
    utils.print_separator("RETRIEVAL PIPELINE READY")
    return rag_chain, vector_store


if __name__ == "__main__":
    rag_chain, _ = run_retrieval_pipeline()
    
    # Test query
    query(rag_chain, "What are the fundamental rights in Nepal's constitution?")