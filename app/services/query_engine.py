# app/services/query_engine.py
class QueryEngine:
    def __init__(self, index):
        self.query_engine = index.as_query_engine()
    
    def query(self, question):
        response = self.query_engine.query(question)
        return str(response)