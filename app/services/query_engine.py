# app/services/query_engine.py
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

class QueryEngine:
    def __init__(self, index):
        self.index = index
        self.query_engine = index.as_query_engine(
            streaming=False,  # Disable streaming for better response formatting
            similarity_top_k=5,  # Increase context window
            response_mode="compact"  # Use compact mode for cleaner responses
        )
    
    def process_response(self, response_text):
        """Clean up response formatting"""
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(response_text.split())
        
        # Fix spacing after punctuation
        cleaned = cleaned.replace('.','. ').replace('!','! ').replace('?','? ')
        
        # Remove any double spaces
        cleaned = ' '.join(cleaned.split())
        
        # Format into readable paragraphs
        sentences = cleaned.split('. ')
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 3:  # Group roughly 3 sentences per paragraph
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
                
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + '.')
            
        # Join paragraphs with double newline for readability
        return '\n\n'.join(paragraphs)
    
    def query(self, question):
        """Query with priority to recent documents"""
        try:
            logger.info(f"Processing query: {question}")
            
            # If it's a summary request, prioritize the latest document
            if any(word in question.lower() for word in ['summary', 'summarize', 'what is this about', 'what is it about']):
                # Access documents using the correct API
                docstore = self.index.storage_context.docstore
                all_docs = list(docstore.docs.values())
                
                # Sort documents by timestamp in filename (newest first)
                all_docs.sort(key=lambda x: x.metadata.get('file_name', ''), reverse=True)
                
                latest_doc = all_docs[0] if all_docs else None
                
                if latest_doc:
                    logger.info(f"Generating summary for latest document: {latest_doc.metadata.get('file_name')}")
                    # Create a summary prompt with context about the latest document
                    summary_prompt = (
                        "Provide a clear and concise summary of the latest document. "
                        "Focus on the main points and key insights. "
                        "Format the response in clean paragraphs without bullet points or excessive line breaks."
                    )
                    response = self.query_engine.query(summary_prompt)
                else:
                    logger.warning("No latest document found for summarization")
                    response = "I couldn't find the latest document to summarize. Please ensure documents are properly uploaded."
            else:
                response = self.query_engine.query(question)
            
            # Clean up response formatting
            response_text = self.process_response(str(response))
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise