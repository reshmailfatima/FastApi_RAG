# app/services/llm_service.py
from google.generativeai import GenerativeModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from app.config import Config

def get_gemini_llm():
    """Configure Gemini Pro LLM"""
    genai.configure(api_key=Config.GOOGLE_API_KEY)
    model = GenerativeModel('models/gemini-1.5-flash')
    return model

def get_huggingface_embeddings():
    """Configure HuggingFace Embeddings using a more powerful model"""
    return SentenceTransformer('BAAI/bge-large-en-v1.5')