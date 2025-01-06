# app/services/document_loader.py
from llama_index.readers.file import PDFReader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_documents(directory_path):
    loader = PDFReader()
    documents = []
    
    # Load all PDF files from the specified directory
    pdf_files = sorted(Path(directory_path).glob("*.pdf"), reverse=True)  # Sort by name in reverse order
    logger.info(f"Found {len(pdf_files)} PDF files, processing in reverse chronological order")
    
    for pdf_file in pdf_files:
        docs = loader.load_data(file=pdf_file)
        # Add metadata about file order
        for doc in docs:
            doc.metadata = {
                'file_name': pdf_file.name,
                'is_latest': pdf_file == pdf_files[0] if pdf_files else False
            }
        documents.extend(docs)
    
    return documents
