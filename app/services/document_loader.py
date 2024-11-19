from llama_index.readers.file import PDFReader
from pathlib import Path

def load_documents(directory_path):
    loader = PDFReader()
    documents = []
    
    pdf_files = Path(directory_path).glob("*.pdf")
    for pdf_file in pdf_files:
        documents.extend(loader.load_data(file=pdf_file))
    
    return documents