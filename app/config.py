# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# app/config.py updates
class Config:
    # Google API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    GEMINI_MODEL_NAME = "models/gemini-1.5-flash"
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Updated to better embedding model