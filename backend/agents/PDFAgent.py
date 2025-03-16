
from chromadb.api.client import Client
from chromadb.utils import embedding_functions


class PDFAgent:
    """Handles retrieval from ChromaDB and passes relevant chunks to LLM."""
    
    def __init__(self,llm, db_name: str,chroma_client: Client):
        """Initialize with a ChromaDB collection."""
        self.collection = chroma_client.get_collection(
            db_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.llm = llm
    
    
    def retrieve(self, query: str):
        """Retrieves relevant documents from ChromaDB."""
        results = self.collection.query(query_texts=[query], n_results=5, include=["documents"])["documents"]
        # print("pdf_agent Results: ", results)
        message = [
            ("system", "You are a helpful assistant who answers only on the basis of the information provided by the user."),
            ("user", f"Only based on the following context answer the question:\n Context:\n   {results}\n        Question: {query}")
        ]
        
        res = self.llm.invoke(message)
        return res