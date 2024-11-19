# app/services/index_creator.py
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .azure_openai_llm import get_azure_llm

def create_embeddings():
    """Create local embeddings"""
    return HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2"
    )

def create_index(documents):
    """Create index with Azure OpenAI LLM and local embeddings"""
    # Configure LLM
    Settings.llm = get_azure_llm()
    
    # Configure embeddings
    embed_model = create_embeddings()
    Settings.embed_model = embed_model
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True
    )
    return index