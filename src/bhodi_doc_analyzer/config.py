import os
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatLlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# TOKENIZER AND LLM INITIALIZATION
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Qwen2.5-Coder-7B-Instruct",
    use_fast=True
)

LOCAL_MODEL = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" # Local model path

llm = ChatLlamaCpp(
    model_path=LOCAL_MODEL,
    temperature=0.1,
    n_ctx=3000,
    n_gpu_layers=-1,
    n_batch=50,
    max_tokens=3000,
    top_p=0.9,
    verbose=False
)

# =============================================================================
# EMBEDDINGS, AND NON-PERSISTENT VECTORSTORE INITIALIZATION (For chat memory)
# =============================================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

# Non-persistent (in-memory) Chroma instance for chatbot conversation
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=None  # In-memory only
)

# Retriever for chat context retrieval (volatile).
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
