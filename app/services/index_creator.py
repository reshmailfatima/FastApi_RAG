# app/services/index_creator.py
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from app.services.llm_service import get_gemini_llm, get_huggingface_embeddings
from app.config import Config
import faiss
import os
import logging

logger = logging.getLogger(__name__)

def create_index(documents, persist_dir="./storage"):
    """Create index with Gemini and HuggingFace services"""
    try:
        logger.info(f"Starting index creation with {len(documents)} documents")
        
        # Configure Gemini LLM
        llm = Gemini(model_name="models/gemini-1.5-flash", api_key=Config.GOOGLE_API_KEY)
        Settings.llm = llm
        logger.info("LLM configured successfully")
        
        # Configure HuggingFace Embeddings
        embed_model = HuggingFaceEmbedding(
            model_name="all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model
        logger.info("Embedding model configured successfully")
        
        # Create storage directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Check if we have a stored index
        if os.path.exists(os.path.join(persist_dir, "faiss.index")) and \
           os.path.exists(os.path.join(persist_dir, "docstore.json")):
            try:
                logger.info("Found existing index, attempting to load...")
                faiss_index = faiss.read_index(os.path.join(persist_dir, "faiss.index"))
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=persist_dir
                )
                index = load_index_from_storage(storage_context)
                logger.info("Successfully loaded existing index")
                return index
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                # If loading fails, we'll create a new index
    
        # Create new index
        dimension = 384  # Dimension for all-MiniLM-L6-v2
        logger.info(f"Creating new FAISS index with dimension {dimension}")
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Create new storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index with progress logging
        logger.info("Starting document indexing...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        logger.info("Document indexing completed successfully")
        
        # Save index data
        try:
            logger.info("Saving index to disk...")
            faiss.write_index(faiss_index, os.path.join(persist_dir, "faiss.index"))
            index.storage_context.persist(persist_dir=persist_dir)
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
        
        return index
        
    except Exception as e:
        logger.error(f"Error in create_index: {e}")
        raise