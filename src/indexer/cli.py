import os
import argparse
from pathlib import Path
from .config_indexer import initialize_vectorstore
from .document_indexing import (
    load_and_index_documents_from_directory,
    load_and_index_single_file,
)

def main() -> None:
    """
    Command line tool for document indexing.
    The vectorstore persistence directory is always set to "chroma_db".
    
    Usage:
        bhodi-index <document_path>
    """
    parser = argparse.ArgumentParser(
        description="Index documents from a specified file or directory using a fixed 'chroma_db' for persistence."
    )
    parser.add_argument(
        "document_path",
        type=Path,
        help="Path (file or directory) containing the documents to vectorize."
    )
    args = parser.parse_args()

    # Resolve the document path.
    document_path = str(args.document_path.resolve())
    # Set the persist directory to a fixed folder "chroma_db" in the current working directory.
    persist_directory = os.path.join(os.getcwd(), "chroma_db")

    # Initialize the vectorstore.
    vectorstore, _ = initialize_vectorstore(persist_directory)

    # Index directly based on the provided document path.
    if os.path.isdir(document_path):
        try:
            count = load_and_index_documents_from_directory(document_path, vectorstore)
            print(f"Indexed {count} document fragments from directory.")
        except Exception as e:
            print(f"Error indexing directory: {e}")
    elif os.path.isfile(document_path):
        try:
            count = load_and_index_single_file(document_path, vectorstore)
            print(f"Indexed {count} document fragments from file.")
        except Exception as e:
            print(f"Error indexing file: {e}")
    else:
        print("Invalid document path provided.")

if __name__ == "__main__":
    main()
