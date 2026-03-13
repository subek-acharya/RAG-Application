"""
Retrieval and generation pipeline.
"""
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import cohere
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import re

import config
import utils
from indexing import get_embeddings, load_vector_store

import warnings
import logging

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Cohere client (only if reranking is enabled)
if config.USE_RERANK and config.COHERE_API_KEY:
    cohere_client = cohere.Client(config.COHERE_API_KEY)
else:
    cohere_client = None


# ------------ pydantic for structured output -----------------
class Citation(BaseModel):
    """A single citation from a source document."""
    source_id: int = Field(description="Source number [1, 2, etc.]")
    page: Optional[str] = Field(default="N/A", description="Page number from the document")
    quote: str = Field(description="The relevant quote or text from the source")


class AnswerWithCitations(BaseModel):
    """Structured answer with citations."""
    answer: str = Field(description="The complete answer to the question")
    citations: List[Citation] = Field(description="List of citations used to support the answer")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")

# ----------- LLM Initialization -------------
def get_llm():
    """Initialize HuggingFace LLM."""
    print(f"Loading LLM: {config.LLM_MODEL}")
    
    if config.USE_4BIT:
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
            bnb_4bit_use_double_quant=True  # Double quantization for more savings
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            quantization_config=quantization_config,
            device_map="auto",  # Automatically place on GPU
            attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        
        # Create pipeline with quantized model
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            return_full_text=False,
            repetition_penalty=1.1,
        )
    else:
        # Standard loading (no quantization)
        pipe = pipeline(
            "text-generation",
            model=config.LLM_MODEL,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.3,
            return_full_text=False,
            repetition_penalty=1.2,
        )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ---------- RETRIEVER AND RERANKING -------------
def get_retriever(vector_store):
    """Create retriever from vector store."""
    # If reranking is enabled, fetch more documents initially
    if config.USE_RERANK:
        k = config.INITIAL_TOP_K
    else:
        k = config.TOP_K
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever

def rerank_documents(query: str, documents: list) -> list:
    """
    Rerank documents using Cohere's rerank model.
    
    Args:
        query: User's question
        documents: List of Document objects from retriever
    
    Returns:
        List of reranked Document objects (top FINAL_TOP_K)
    """
    if not cohere_client or not documents:
        return documents[:config.FINAL_TOP_K]
    
    # Extract text from documents
    doc_texts = [doc.page_content for doc in documents]
    
    try:
        # Call Cohere rerank API
        response = cohere_client.rerank(
            model=config.RERANK_MODEL,
            query=query,
            documents=doc_texts,
            top_n=config.FINAL_TOP_K,
            return_documents=False
        )
        
        # Reorder documents based on rerank results
        reranked_docs = []
        for result in response.results:
            doc = documents[result.index]
            doc.metadata["rerank_score"] = result.relevance_score
            reranked_docs.append(doc)
        
        # print(f"  Reranked {len(documents)} docs → Top {len(reranked_docs)}")
        return reranked_docs
    
    except Exception as e:
        print(f"  Reranking failed: {e}. Using original order.")
        return documents[:config.FINAL_TOP_K]

# ----------- DOCUMENT FORMATTING WITH SOURCES  -------------
def format_docs_with_sources(docs):
    """Format documents with source numbers and page info for citation."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        
        formatted.append(
            f"[Source {i}] (Page {page}):\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted)

def retrieve_and_rerank(inputs):
    """
    Retrieve documents and optionally rerank them.
    """
    question = inputs["question"]
    documents = inputs["documents"]
    
    if config.USE_RERANK:
        documents = rerank_documents(question, documents)
    
    context = format_docs_with_sources(documents)
    
    return {"context": context, "question": question}

# -------- RAG CHAIN WITH CITATIONS ---------------
def create_citation_prompt():
    """Create prompt template for structured citation output."""

    parser = PydanticOutputParser(pydantic_object=AnswerWithCitations)

    format_example = '''{
      "answer": "The fundamental rights include the right to equality, right to freedom, right against exploitation, and right to constitutional remedies. Every citizen has the right to obtain quality goods and services.",
      "citations": [
        {"source_id": 1, "page": "27", "quote": "Every consumer shall have the right to obtain quality goods and services."},
        {"source_id": 2, "page": "21", "quote": "Every person shall have the right against exploitation."}
      ],
      "confidence": "high"
    }'''
    
    template = """[INST] You are a legal document assistant. Answer questions based on the provided sources.
    
    IMPORTANT RULES:
    1. Respond with ONLY valid JSON - no other text
    2. The "answer" field must be a COMPLETE, DETAILED answer in full sentences (not just article numbers)
    3. Explain the content, don't just list numbers
    4. Include relevant citations with exact quotes
    
    Sources:
    {context}
    
    Question: {question}
    
    Required JSON format:
    {format_example}
    
    Remember:
    - Write a FULL, DETAILED answer explaining the content (2-4 sentences minimum)
    - Do NOT just list article numbers - explain what the rights ARE
    - Include citations with page numbers and exact quotes
    - Set confidence: "high", "medium", or "low"
    
    Respond with JSON only: [/INST]"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"format_example": format_example}
    )

    return prompt, parser

def parse_llm_output(output: str, parser: PydanticOutputParser) -> AnswerWithCitations:
    """Parse LLM output to structured format with fallback."""
    # The prompt ends with '{' to force JSON; prepend it back
    if not output.strip().startswith("{"):
        output = "{" + output

    # Fix common malformed keys like "source_id someword" -> "source_id"
    output_clean = re.sub(r'"source_id\s+\w+"', '"source_id"', output)

    try:
        # Try direct parsing on cleaned output
        return parser.parse(output_clean)
    except Exception:
        pass

    # Try to extract and parse JSON block
    for candidate in [output_clean, output]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', candidate)
            if json_match:
                data = json.loads(json_match.group())
                return AnswerWithCitations(**data)
        except:
            pass

    # Extract answer and confidence via regex as last resort
    answer_match = re.search(r'"answer"\s*:\s*"([\s\S]*?)"(?=\s*,|\s*\})', output)
    confidence_match = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', output, re.IGNORECASE)
    answer = answer_match.group(1) if answer_match else output.strip()
    confidence = confidence_match.group(1).lower() if confidence_match else "low"
    return AnswerWithCitations(answer=answer, citations=[], confidence=confidence)

def create_rag_chain(retriever, llm):
    """Create the RAG chain with structured citations."""
    prompt, parser = create_citation_prompt()
    
    rag_chain = (
            {"documents": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(retrieve_and_rerank) 
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return rag_chain, parser

# ---------- QUERY FUNCTIONS -------------
def query_with_citations(rag_chain, parser, question):
    """
    Query the RAG chain and return structured response with citations.
    
    Returns:
        AnswerWithCitations object with answer, citations, and confidence
    """
    print(f"\nQuestion: {question}")
    
    # Get raw output from chain
    raw_output = rag_chain.invoke(question)
    
    # Parse to structured format
    result = parse_llm_output(raw_output, parser)
    
    return result


def print_answer_with_citations(result: AnswerWithCitations):
    """Pretty print the answer with citations."""
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result.answer)
    
    print("\n" + "-"*60)
    print(f"CONFIDENCE: {result.confidence.upper()}")
    
    if result.citations:
        print("\n" + "-"*60)
        print("CITATIONS:")
        print("-"*60)
        for i, citation in enumerate(result.citations, 1):
            print(f"\n  [{citation.source_id}] Page {citation.page}:")
            print(f"      \"{citation.quote[:200]}{'...' if len(citation.quote) > 200 else ''}\"")
    else:
        print("\n  No citations provided.")
    
    print("\n" + "="*60)

def query(rag_chain, parser, question):
    """Query and display formatted response."""
    result = query_with_citations(rag_chain, parser, question)
    print_answer_with_citations(result)
    return result

# ----- PIPELINE INITIALIZATION ---------
def run_retrieval_pipeline():
    """Initialize retrieval components."""
    utils.print_separator("INITIALIZING RETRIEVAL PIPELINE")
    
    # Load embeddings and vector store
    embeddings = get_embeddings()
    vector_store = load_vector_store(embeddings)
    
    # Create retriever
    retriever = get_retriever(vector_store)

    # Print rerank status
    if config.USE_RERANK:
        print(f"Reranking: ENABLED (model: {config.RERANK_MODEL})")
        print(f"  Initial retrieval: {config.INITIAL_TOP_K} docs")
        print(f"  After reranking: {config.FINAL_TOP_K} docs")
    else:
        print(f"Reranking: DISABLED")
        print(f"  Retrieval: {config.TOP_K} docs")
    
    # Load LLM
    llm = get_llm()
    
    # Create RAG chain with citation parser
    rag_chain, parser = create_rag_chain(retriever, llm)
    
    utils.print_separator("RETRIEVAL PIPELINE READY")
    return rag_chain, parser, vector_store


if __name__ == "__main__":
    rag_chain, parser, _ = run_retrieval_pipeline()
    
    # Test query
    query(rag_chain, parser, "What are the fundamental rights in Nepal's constitution?")