from typing import Any, List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_index_documents_from_directory(directory_path: str, vectorstore: Any) -> int:
    """
    Loads documents from a directory, splits them and adds them to the vectorstore.
    
    Args:
        directory_path (str): Path to the directory containing documents.
        vectorstore (Any): The vectorstore instance where documents will be added.
    
    Returns:
        int: The number of document fragments indexed.
    """
    loaders = [
        DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ),
        DirectoryLoader(
            directory_path,
            glob="**/*.{txt,md,py,js,ts,cpp,java,go,rs}",
            loader_cls=TextLoader
        )
    ]
    
    documents: List = []
    for loader in loaders:
        documents.extend(loader.load())


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    vectorstore.add_documents(split_docs)
    
    return len(split_docs)

def load_and_index_single_file(file_path: str, vectorstore: Any) -> int:
    """
    Loads a single file, splits its contents and adds it to the vectorstore.
    
    Args:
        file_path (str): Path to the file.
        vectorstore (Any): The vectorstore instance where the document will be added.
    
    Returns:
        int: The number of document fragments indexed.
    """
    # Select the loader based on file extension.
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)


    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    vectorstore.add_documents(split_docs)
    
    return len(split_docs)
