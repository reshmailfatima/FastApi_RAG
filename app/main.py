# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from pathlib import Path
import os
import asyncio

from .services.document_loader import load_documents
from .services.index_creator import create_index
from .services.query_engine import QueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document QA System", description="RAG-based Question Answering System")

# Pydantic models for request/response
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    error: Optional[str] = None

# Global variables to store the query engine
query_engine = None

async def initialize_rag_system():
    """Async function to initialize RAG system"""
    global query_engine
    try:
        # Ensure data directory exists
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created directory: {data_dir}")
            return None

        # Check for PDF files
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in the data directory.")
            return None

        # Load documents
        logger.info("Loading documents...")
        documents = load_documents(data_dir)
        
        if not documents:
            logger.warning("No documents were successfully loaded.")
            return None

        # Create index
        logger.info("Creating index...")
        index = create_index(documents)
        
        # Initialize query engine
        logger.info("Initializing query engine...")
        query_engine = QueryEngine(index)
        
        logger.info("RAG system initialized successfully!")
        return query_engine

    except Exception as e:
        logger.error(f"Startup error: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event"""
    global query_engine
    try:
        query_engine = await initialize_rag_system()
    except Exception as e:
        logger.error(f"Unexpected startup error: {e}")

@app.post("/query")
async def query_documents(question: Question):
    """Submit a question about the loaded documents"""
    global query_engine
    
    if query_engine is None:
        # Attempt to reinitialize if query_engine is None
        query_engine = await initialize_rag_system()
        
        if query_engine is None:
            return Answer(
                answer="",
                error="RAG system not initialized. Please add documents to the 'data' directory."
            )
    
    try:
        response = query_engine.query(question.text)
        return Answer(answer=response)
    except Exception as e:
        logger.error(f"Query error: {e}")
        return Answer(
            answer="",
            error=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Check if the system is ready to handle queries"""
    global query_engine
    
    # Attempt to initialize if not already done
    if query_engine is None:
        query_engine = await initialize_rag_system()
    
    return {
        "status": "healthy" if query_engine is not None else "not_ready",
        "message": "System is ready for queries" if query_engine is not None else "System is initializing"
    }

# Optional: Ensure a clean shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down the application")