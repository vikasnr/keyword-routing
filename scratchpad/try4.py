import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
# from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from llm import get_llm
import warnings
import shutil
import time
warnings.filterwarnings("ignore")

# from db_handler import query_vector_1, query_vector_2, query_vector_3, route_query
# Initialize FastAPI
app = FastAPI()

llm = get_llm()

# Delete chromadb folder if it exists
chromadb_path = "chromadb"
if os.path.exists(chromadb_path) and os.path.isdir(chromadb_path):
    print("Deleting existing chromadb folder...")
    shutil.rmtree(chromadb_path)
    print("Sleeping for 5 seconds...")
    time.sleep(5)
    

# Initialize Vector Stores with HuggingFace Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store_1 = Chroma(persist_directory="chromadb/db_pdf1", embedding_function=embeddings)
vector_store_2 = Chroma(persist_directory="chromadb/db_pdf2", embedding_function=embeddings)
vector_store_3 = Chroma(persist_directory="chromadb/db_pdf3", embedding_function=embeddings)

# Function to process PDF files and store embeddings
def process_pdf_files(pdf_file: str, vector_store: Chroma):
    
    loader = PyMuPDFLoader(pdf_file)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    vector_store.add_documents(split_docs)

# Process PDF files and store embeddings in vector stores
def process_pdfs(pdf_files: List[str], vector_stores: List[Chroma]):
    for pdf_file, vector_store in zip(pdf_files, vector_stores):
        process_pdf_files(pdf_file, vector_store)

pdf_files = {
    "/home/vikasnr/codebase/crsl/pdfs/crazyones-pdfa.pdf": vector_store_1,
    "/home/vikasnr/codebase/crsl/pdfs/midterm_report.pdf": vector_store_2,
    "/home/vikasnr/codebase/crsl/pdfs/spores_manual.pdf": vector_store_3
}


process_pdfs(list(pdf_files.keys()), list(pdf_files.values()))

# Define Tools for Vector Store Queries
def query_vector_store(vector_store, query: str) -> str:
    docs = vector_store.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])

query_vector_1 = Tool(
    name="Query PDF 1",
    func=lambda query: query_vector_store(vector_store_1, query),
    description="Search PDF 1 vector store for answers."
)

query_vector_2 = Tool(
    name="Query PDF 2",
    func=lambda query: query_vector_store(vector_store_2, query),
    description="Search PDF 2 vector store for answers."
)

query_vector_3 = Tool(
    name="Query PDF 3",
    func=lambda query: query_vector_store(vector_store_3, query),
    description="Search PDF 3 vector store for answers."
)

# Define Agents
memory = ConversationBufferMemory(memory_key="chat_history")

manager_prompt = PromptTemplate(
    input_variables=["query"],
    template="You are an assistant that strictly answers only from the provided documents. If the information is not found in the documents, simply say: 'I'm sorry, but I couldn't find relevant information in the provided documents.' Do not use any external knowledge. Do not rely on general knowledge.\n\nQuery: {query}"
)

manager_chain = LLMChain(llm=llm, prompt=manager_prompt, memory=memory)

manager_agent = initialize_agent(
    tools=[query_vector_1, query_vector_2, query_vector_3],
    llm=LLMChain,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Define Response Aggregation Agent
def aggregate_responses(responses: List[str]) -> str:
    combined_response = "\n".join(responses)
    return f"{combined_response}"

# Define API Request Model
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        query = request.query
        agent_tools = [query_vector_1,query_vector_2,query_vector_3]
        agent_executor = initialize_agent(
            tools=agent_tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        responses = [agent_executor.run(query) for tool in agent_tools]
        final_response = aggregate_responses(responses)
        return {"query": query, "response": final_response}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



# import os
# import json
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# from db_handler import route_query
# from pdf_agents import execute_agent

# # Initialize FastAPI
# app = FastAPI()

# # Define API Request Model
# class ChatRequest(BaseModel):
#     query: str

# @app.post("/chat")
# def chat(request: ChatRequest):
#     try:
#         query = request.query
#         final_response = execute_agent(query)
#         return {"query": query, "response": final_response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
