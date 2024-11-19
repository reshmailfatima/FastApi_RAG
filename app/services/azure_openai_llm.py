# app/services/azure_openai_llm.py
from llama_index.llms.azure_openai import AzureOpenAI
from ..config import Config
import os

def get_azure_llm():
    return AzureOpenAI(
        model="gpt-4o-mini",
        engine="gpt4omini",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1
    )