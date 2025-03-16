from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Crew, Task
from sentence_transformers import SentenceTransformer
import chromadb
import openai

import requests
import json

def query_llm_rest_pix(mlist, image=None):
    url = 'http://<masked>:9109/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer token'
    }
    data = {
        "model": "mistralai/Pixtral-12B-2409"
    }
    data.update({"messages": mlist})

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 400:
        return "I'm unsure about the answer. Please provide more context or ask a different question."
    else:
        return response.json()["choices"][0]['message']['content']

# Initialize FastAPI app
app = FastAPI()

# Initialize Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB clients for each PDF
CHROMA_DB_PATH = "/home/vikasnr/codebase/crsl/pdf_agents/chromadb"

chroma_clients = {
    "pdf1": chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/insurance"),
    "pdf2": chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/rivers"),
    "pdf3": chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/wildlife")
}

class ChatRequest(BaseModel):
    query: str

# Define PDF-specific agents
class PDFQueryAgent(Agent):
    def __init__(self, pdf_name):
        super().__init__(name=f"Bot {pdf_name.upper()}")
        self.pdf_name = pdf_name
        self.chroma_client = chroma_clients[pdf_name]
        self.collection = self.chroma_client.get_or_create_collection("documents")

    def search_vector_db(self, query):
        query_embedding = embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
        return " ".join(results["documents"][0]) if results["documents"] else "No relevant information found."

    def execute(self, query):
        context = self.search_vector_db(query)
        return query_llm_rest_pix([
                {"role": "system", "content": "Answer based only on the provided context."},
                {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
            ]
        )

# Define Manager Agent
class ManagerAgent(Agent):
    def __init__(self, pdf_agents):
        super().__init__(name="Manager Agent")
        self.pdf_agents = pdf_agents

    def route_query(self, query):
        agent_responses = {}
        for pdf_name, agent in self.pdf_agents.items():
            response = agent.execute(query)
            if response and "No relevant information found." not in response:
                agent_responses[pdf_name] = response
        return agent_responses

# Define Response Aggregation Agent
class ResponseAggregatorAgent(Agent):
    def __init__(self):
        super().__init__(name="Aggregator Agent")

    def aggregate(self, responses):
        if not responses:
            return "No relevant answer found across all documents."
        combined_response = "\n".join([f"{key}: {value}" for key, value in responses.items()])
        return query_llm_rest_pix([
                {"role": "system", "content": "Summarize and consolidate the following responses."},
                {"role": "user", "content": combined_response}
            ]
        )

# Initialize agents
pdf_agents = {
    "pdf1": PDFQueryAgent("pdf1"),
    "pdf2": PDFQueryAgent("pdf2"),
    "pdf3": PDFQueryAgent("pdf3")
}
manager_agent = ManagerAgent(pdf_agents)
aggregator_agent = ResponseAggregatorAgent()

responses = manager_agent.route_query("What is the capital of France?")
final_response = aggregator_agent.aggregate(responses)
# Define API endpoint
# @app.post("/chat")
# def chat(request: ChatRequest):
#     try:
#         responses = manager_agent.route_query(request.query)
#         final_response = aggregator_agent.aggregate(responses)
#         return {"response": final_response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
