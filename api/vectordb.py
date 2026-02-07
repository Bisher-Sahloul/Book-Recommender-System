import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

# Paths
persistent_directory = os.path.join('artifacts', "vectorstores" , "chroma_db")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
def get_vector_db():
    if not os.path.exists(persistent_directory):
        raise FileNotFoundError("Vector store does not exist. Please create it first.")
    return Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )

# Main search function for API
def search_books(
    query: str,
    search_type: str = "similarity",
    k: int = 5
) -> List[Dict]:
    """
    Perform semantic search in vector DB and return book metadata.
    """
    db = get_vector_db()
    recs = db.similarity_search(query, k = k)
    return [rec.metadata for rec in recs]
