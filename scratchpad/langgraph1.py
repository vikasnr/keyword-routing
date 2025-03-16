from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph,START,ChatState



from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langgraph.agent import Agent
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

from PyPDF2 import PdfReader
from llm import get_llm
# app = FastAPI()

# Initialize components
llm = get_llm()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")



import requests
import json

def query_llm_rest_pix(mlist, image=None):
    url = 'http://172.19.104.12:9109/v1/chat/completions'
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



# Define input schema
class QueryRequest(BaseModel):
    query: str

def chunk_and_store(pdf_path: str, db_name: str, chunk_size: int = 500):
    """Chunks, embeds, and stores PDF content into ChromaDB."""
    # reader = PdfReader(pdf_path)
    # text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    # chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # embeddings = embedder.encode(chunks).tolist()

    documents = []
    print("pdf:", pdf_path)
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)

    chunks = text_splitter.split_documents(documents)


    # text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)

    # docs = text_splitter.split_documents(documents)
    collection = chroma_client.get_or_create_collection(db_name,embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))

    collection.add(ids=[f"doc_{i}" for i in range(len(chunks))], documents=[i.page_content for i in chunks],metadatas=[i.metadata for i in chunks])
    # for i, chunk in enumerate(chunks):
    #     collection.add(ids=f"doc_{i}",documents=[{"text": chunk}], embeddings=[embeddings[i]])


pdf1 = "insurance"
pdf2 = "rivers"
pdf3 = "wildlife"


# Load PDFs
pdf_files = {pdf1: f"/home/vikasnr/codebase/crsl/pdfs/{pdf1}.pdf", 
             pdf2: f"/home/vikasnr/codebase/crsl/pdfs/{pdf2}.pdf", 
             pdf3: f"/home/vikasnr/codebase/crsl/pdfs/{pdf3}.pdf"}

for db_name, pdf_path in pdf_files.items():
    chunk_and_store(pdf_path, db_name)

# Define Agents
def router_agent(query: str):
    """Determines which PDF agent(s) should process the query."""
    # Simple logic: Route to all agents (can be improved)
    return ["agent_a", "agent_b", "agent_c"]

def pdf_agent(query: str, db_name: str):
    """Handles retrieval from ChromaDB and passes relevant chunks to LLM."""
    collection = chroma_client.get_or_create_collection(db_name,embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))

    results = collection.query(query_texts=[query], n_results=5)
    print("pdf_agent Results: ", results)
    retrieved_texts = [doc["text"] for doc in results]
    response = query_llm_rest_pix([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer based on:        {retrieved_texts}\n        Question: {query}"}
    ])
    return response

def aggregator_agent(responses: list):
    """Combines multiple agent responses into one coherent reply."""
    combined_response = "\n".join(responses)
    return combined_response

# Define Multi-Agent Graph
workflow = StateGraph(ChatState)
# workflow.support_multiple_edges = True
workflow.add_node("router", router_agent)
workflow.add_node("agent_a", lambda q: pdf_agent(q, pdf1))
workflow.add_node("agent_b", lambda q: pdf_agent(q, pdf2))
workflow.add_node("agent_c", lambda q: pdf_agent(q, pdf3))
workflow.add_node("aggregator", aggregator_agent)

workflow.add_edge(START, "router")


# Define edges
# workflow.add_edge("router", "agent_a")
# workflow.add_edge("router", "agent_b")
# workflow.add_edge("router", "agent_c")
# workflow.add_edge("agent_a", "aggregator")
# workflow.add_edge("agent_b", "aggregator")
# workflow.add_edge("agent_c", "aggregator")

compiled_workflow = workflow.compile()
response = compiled_workflow.run("what is the claim process?")
print(response)

# @app.post("/chat")
# def chat(request: QueryRequest):
#     """Handles user query and orchestrates agent interactions."""
#     try:
#         response = compiled_workflow.run(request.query)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
