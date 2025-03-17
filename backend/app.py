
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chromadb import HttpClient
from chromadb.utils import embedding_functions

from PyPDF2 import PdfReader

from agents.PDFAgent import PDFAgent
from agents.AggregatorAgent import AggregatorAgent
from agents.ManagerAgent import ManagerAgent
from agents.KeywordDictionary import KeywordDictionary
from llm import get_llm

llm = get_llm()
#define chroma client
chroma_client = HttpClient(host="localhost", port=9158)


extract_keywords = False
# Function to load, chunk, and store PDFs into vector stores
def chunk_and_store(pdf_path: str, db_name: str,):
    # Load PDF
    documents = []
    # print("pdf:", pdf_path)
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    chunks = text_splitter.split_documents(documents)
    combined_chunks = " ".join([i.page_content for i in chunks])

    collection = chroma_client.get_or_create_collection(db_name,embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"))

    collection.add(ids=[f"doc_{i}" for i in range(len(chunks))], documents=[i.page_content for i in chunks],metadatas=[i.metadata for i in chunks])

    print(f"Stored {len(chunks)} chunks in vector store.")

    if extract_keywords:
        kd = KeywordDictionary(file_path="keywords.json")
        kd.extract_keywords(llm,db_name, combined_chunks)

pdf1 = "insurance"
pdf2 = "rivers"
pdf3 = "wildlife"

pdfs_folder = '../pdfs'

# Load PDFs
pdf_files = {pdf1: f"{pdfs_folder}/{pdf1}.pdf", 
             pdf2: f"{pdfs_folder}/{pdf2}.pdf", 
             pdf3: f"{pdfs_folder}/{pdf3}.pdf"}

# Chunk and store PDFs
for db_name, pdf_path in pdf_files.items():
    existing_collections = chroma_client.list_collections()
    if db_name in [col.name for col in existing_collections]:
        print(f"Collection {db_name} already exists. Skipping...")
        continue
    else:
        chunk_and_store(pdf_path, db_name)

#get keywords
kd = KeywordDictionary(file_path="keywords.json")
keywords = kd.get_keywords()


# Define input schema
class QueryRequest(BaseModel):
    query: str

app = FastAPI()


# FastAPI Endpoint for Question Answering
@app.post("/chat")
def query_pdf(request: QueryRequest):
        
    try:
        question = request.query

        # manager agent finding appropriate bots
        manager_agent = ManagerAgent(llm,keywords)
        bots = manager_agent.route(question)

        print("bots selected",bots)

        # bots answering the question
        bot_responses = []
        for bot in bots:
            Bot_X = PDFAgent(llm,bot, chroma_client)
            
            bot_responses.append(Bot_X.retrieve(question).content)
        
        # aggregator agent aggregating the responses
        aggregator_agent = AggregatorAgent(llm)
        final_answer = aggregator_agent.aggregate(bot_responses)

        return {"response": final_answer}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


