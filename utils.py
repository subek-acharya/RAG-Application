"""
Utility functions for the RAG application.
"""

import os


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def print_separator(title=""):
    """Print a separator line for better output readability."""
    print("\n" + "=" * 60)
    if title:
        print(f" {title}")
        print("=" * 60)


def format_documents(docs):
    """Format retrieved documents for display."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"\n--- Document {i} ---")
        formatted.append(f"Content: {doc.page_content[:500]}...")
        if doc.metadata:
            formatted.append(f"Source: {doc.metadata}")
    return "\n".join(formatted)