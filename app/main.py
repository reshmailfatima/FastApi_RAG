# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import shutil
from pathlib import Path
import os
import logging
from pydantic import BaseModel

from app.services.document_loader import load_documents
from app.services.query_engine import QueryEngine
from app.services.index_creator import create_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")

# Global variables
query_engine = None

# Pydantic models
class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# In app/main.py, update the initialize_rag_system function

def initialize_rag_system():
    """Initialize or reinitialize the RAG system"""
    global query_engine
    
    try:
        # Ensure directories exist
        for directory in [DATA_DIR, STORAGE_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

        # Check for PDF files in the correct data directory
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        if not pdf_files:
            logger.info(f"No PDF files found in the data directory: {DATA_DIR}")
            return False

        # Load documents from the correct path
        logger.info(f"Loading documents from: {DATA_DIR}")
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = load_documents(DATA_DIR)
        
        if not documents:
            logger.info("No documents were successfully loaded.")
            return False
            
        logger.info(f"Successfully loaded {len(documents)} documents")
        
        # Log first few characters of each document for verification
        for i, doc in enumerate(documents):
            preview = str(doc.text)[:100] if hasattr(doc, 'text') else "No text available"
            logger.info(f"Document {i} preview: {preview}...")

        logger.info(f"Creating and persisting index in: {STORAGE_DIR}")
        index = create_index(documents, persist_dir=STORAGE_DIR)
        
        # Initialize query engine
        logger.info("Initializing query engine...")
        query_engine = QueryEngine(index)
        
        # Test query to verify system is working
        test_query = "What is this document about?"
        try:
            test_response = query_engine.query(test_query)
            logger.info(f"Test query response: {test_response[:100]}...")
        except Exception as e:
            logger.error(f"Test query failed: {e}")
        
        return True

    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

@app.post("/query/", response_model=QueryResponse)
async def query_documents(query: Query):
    """Query the RAG system with better error handling"""
    global query_engine

    if not query_engine:
        # Try to initialize if not already done
        if not initialize_rag_system():
            raise HTTPException(status_code=400, detail="System not initialized. Please upload documents first.")

    try:
        logger.info(f"Processing query: {query.question}")
        answer = query_engine.query(query.question)
        
        if not answer or "not provided in the context" in str(answer).lower():
            logger.warning("Query returned no results or insufficient context")
            # You might want to return a more helpful message here
            return QueryResponse(answer="I couldn't find specific information about that in the documents. Could you rephrase your question or be more specific?")
            
        return QueryResponse(answer=str(answer))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your query")

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    if initialize_rag_system():
        logger.info("RAG system initialized successfully")
    else:
        logger.info("RAG system waiting for documents")

@app.post("/upload/", response_model=dict)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload PDF documents to the system with timestamp prefix for ordering"""
    try:
        uploaded_files = []
        from datetime import datetime
        
        # Save uploaded files to the data directory with timestamp prefix
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Generate timestamp prefix in reverse chronological format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_filename = f"{timestamp}_{file.filename}"
            
            file_path = os.path.join(DATA_DIR, timestamped_filename)
            
            try:
                # Remove older versions of the same file if they exist
                for existing_file in os.listdir(DATA_DIR):
                    if file.filename in existing_file:
                        os.remove(os.path.join(DATA_DIR, existing_file))
                
                # Save new file with timestamp
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(timestamped_filename)
                logger.info(f"Successfully uploaded: {timestamped_filename} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}")

        # Reinitialize the RAG system
        if initialize_rag_system():
            return {
                "message": f"Successfully uploaded {len(files)} documents and initialized the system",
                "uploaded_files": uploaded_files,
                "data_directory": DATA_DIR
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize system with uploaded documents")

    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def query_documents(query: Query):
    """Query the RAG system"""
    global query_engine

    if not query_engine:
        raise HTTPException(status_code=400, detail="System not initialized. Please upload documents first.")

    try:
        answer = query_engine.query(query.question)
        return QueryResponse(answer=str(answer))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing your query")

@app.get("/status/")
async def get_status():
    """Get the system status with sorted document list"""
    try:
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        # Sort files by name in reverse order (since timestamp is prefix, this will show newest first)
        pdf_files.sort(reverse=True)
        has_documents = len(pdf_files) > 0
        is_initialized = query_engine is not None

        # Remove timestamp prefix from displayed names
        document_names = []
        for f in pdf_files:
            # Split on underscore and take everything after the timestamp
            original_name = '_'.join(f.name.split('_')[2:]) if f.name.count('_') >= 2 else f.name
            document_names.append(original_name)

        return {
            "status": "ready" if is_initialized else "waiting_for_documents",
            "documents_loaded": len(pdf_files),
            "document_names": document_names,  # Show original filenames without timestamps
            "system_initialized": is_initialized,
            "data_directory": DATA_DIR,
            "storage_directory": STORAGE_DIR
        }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking system status")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
