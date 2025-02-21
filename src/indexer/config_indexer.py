import os
from typing import Tuple, Any
from langchain_chroma import Chroma
from bhodi_doc_analyzer.config import embeddings

def initialize_vectorstore(persist_directory: str) -> Tuple[Any, Any]:
    """
    Initializes the embeddings, vectorstore, and retriever for document indexing.
    Uses persistent storage.
    
    Args:
        persist_directory (str): Path where the vectorstore will persist data.
    
    Returns:
        Tuple containing the vectorstore and retriever.
    """
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return vectorstore, retriever

# Create a global persistent instance using a fixed folder (e.g. "chroma_db")
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
persistent_vectorstore, persistent_retriever = initialize_vectorstore(PERSIST_DIRECTORY)
