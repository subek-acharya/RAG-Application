# RAG Application

A Retrieval-Augmented Generation (RAG) application with **Cohere Reranking** and **Structured Citations** that enables intelligent question-answering on Nepal's Constitution using LangChain, Mistral-7B, and ChromaDB.

## Overview

This project implements a production-ready RAG pipeline that:
- Processes PDF documents into searchable embeddings
- Uses **Cohere Reranking** for improved document relevance
- Generates answers using **Mistral-7B** (4-bit quantized)
- Provides **structured JSON output** with source citations, page numbers, and confidence levels

## System Architecture

![RAG System Architecture](./system_architecture.png)

The system follows a two-phase architecture:

1. **Indexing Pipeline**: PDF → Chunks → Embeddings → Vector Database
2. **Query Pipeline**: Question → Retrieve → Context + Prompt → LLM → Answer

## Technologies Used

| Component | Technology | Description |
|-----------|------------|-------------|
| Framework | LangChain | Orchestration and chain management |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim dense vectors |
| Vector Database | ChromaDB | Persistent vector storage |
| Reranking | Cohere rerank-v3.5 | Semantic relevance reordering |
| LLM | Mistral-7B-Instruct-v0.3 | 4-bit quantized for efficiency |
| Output Parsing | Pydantic | Structured JSON validation |

## Project Structure

```bash
RAG_APPLICATION/
│
├── data/
│   ├── documents/
│   │   └── Constitution_of_Nepal.pdf    # Source document
│   └── chroma_db/                       # Persisted vector store
│
├── config.py              # Configuration settings and paths
├── indexing.py            # Document processing and embedding pipeline
├── retrieval.py           # RAG chain creation and query handling
├── main.py                # Entry point and interactive interface
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Configuration
Update settings in `config.py`:
```python
# LLM Settings
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
USE_4BIT = True  # 4-bit quantization for GPU efficiency

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranking (Cohere)
USE_RERANK = True
RERANK_MODEL = "rerank-v3.5"
INITIAL_TOP_K = 15  # Documents to retrieve
FINAL_TOP_K = 5     # Documents after reranking

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## Setup
### Clone and Install
```bash
git clone https://github.com/subek-acharya/RAG-Application.git
cd RAG-Application

python -m venv rag_env
source rag_env/bin/activate

pip install -r requirements.txt
```
### Setup API Keys
Create a .env file
```env
COHERE_API_KEY=your-cohere-api-key-here
```
Get your free Cohere API key at cohere.com
### Add your document
Place your PDF in `data/documents/` and update `config.py`:
```python
# Change this to your PDF filename
PDF_PATH = os.path.join(DOCUMENTS_DIR, "your_document.pdff")
```
### Run
```bash
python main.py
```
The system will:

Load/create the vector store
Initialize the retrieval pipeline with Cohere reranking
Load Mistral-7B (4-bit quantized)
Enter interactive Q&A mode

## Sample Output

```bash
============================================================
 TESTING RAG SYSTEM WITH CITATIONS
============================================================

Question: What is the voting age in Nepal?

============================================================
ANSWER:
============================================================
In Nepal, every citizen who resides within the territory of the state 
and has completed the age of eighteen years shall have a right to vote 
in any one election constituency, as per the laws.

------------------------------------------------------------
CONFIDENCE: HIGH

------------------------------------------------------------
CITATIONS:
------------------------------------------------------------

  [1] Page 114:
      "Each citizen of Nepal who resides within the territory of the 
       State and who has completed the age of eighteen years shall 
       have a right to vote in any one election constituency in 
       accordance with law."

  [2] Page 59:
      "Each citizen of Nepal who has completed the age of eighteen 
       years shall have the right to vote in any one election 
       constituency as provided for in the Federal law."

  [3] Page 140:
      "Every person who has completed the age of eighteen years and 
       whose name is included in the electoral rolls of the Municipality 
       shall have a right to vote as provided for in the Federal law."

  [4] Page 134:
      "A person who has completed the age of twenty one years, shall 
       be qualified to be elected to the office of the Chairperson, 
       Vice-Chairperson, Ward Chairperson and member."

  [5] Page 136:
      "A person who has completed the age of eighteen years and whose 
       name is included in the electoral rolls of the Municipality, 
       shall be qualified to be elected to the office of the Mayor, 
       Deputy Mayor, W..."

============================================================
```
## How Reranking works?
```css
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Question: "What is the voting age?"                                   │
│                       │                                                 │
│                       ▼                                                 │
│   Vector Search: Retrieve 15 documents by embedding similarity          │
│   [Doc1, Doc2, Doc3, ..., Doc15]                                        │
│                       │                                                 │
│                       ▼                                                 │
│   Cohere Rerank: Re-score by semantic relevance to query                │
│   [Doc7(0.95), Doc3(0.89), Doc12(0.82), Doc1(0.78), Doc9(0.71)]         │
│                       │                                                 │
│                       ▼                                                 │
│   Top 5 documents sent to LLM with source labels                        │
│   [Source 1] Page 114: Doc7 content...                                  │
│   [Source 2] Page 59:  Doc3 content...                                  │
│   ...                                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
## Structured Ouput Format
```json
{
  "answer": "In Nepal, every citizen who resides within the territory of the state and has completed the age of eighteen years shall have a right to vote in any one election constituency, as per the laws.",
  "citations": [
    {
      "source_id": 1,
      "page": "114",
      "quote": "Each citizen of Nepal who has completed the age of eighteen years..."
    },
    {
      "source_id": 2,
      "page": "59",
      "quote": "Each citizen of Nepal who has completed the age of eighteen years..."
    }
  ],
  "confidence": "high"
}
```

*Developed and tested on **NVIDIA A100-PCIE-40GB** (40GB VRAM, CUDA 13.0)*
