from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import PyPDF2
from crewai import Agent, Task, Crew
import os
from utils.llm import get_openai_llm


app = FastAPI()

# Configuration
PDF_PATHS = {
    1: "/home/vikasnr/codebase/crsl/pdfs/pdf1.pdf",
    2: "/home/vikasnr/codebase/crsl/pdfs/pdf2.pdf",
    3: "/home/vikasnr/codebase/crsl/pdfs/pdf3.pdf"
}

PDF_DESCRIPTIONS = """
1: Description of PDF 1
2: Description of PDF 2
3: Description of PDF 3
"""

class VectorStore:
    def __init__(self, pdf_path, collection_name):
        self.client = HttpClient(host="localhost", port=9158)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.pdf_path = pdf_path
        self.load_pdf()

    def load_pdf(self):
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_chunks = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                # Simple text chunking - you might want to implement more sophisticated chunking
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                text_chunks.extend(chunks)

            embeddings = self.encoder.encode(text_chunks)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=text_chunks,
                ids=[f"id_{i}" for i in range(len(text_chunks))]
            )

    def query(self, question, n_results=3):
        question_embedding = self.encoder.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        return results['documents'][0]

class PDFAgent:
    def __init__(self, pdf_path, collection_name):
        self.vector_store = VectorStore(pdf_path, collection_name)
        self.agent = Agent(
            role='PDF Expert',
            goal='Provide accurate answers based on PDF content',
            backstory='I am an expert in analyzing and answering questions based on PDF content',
            allow_delegation=False,
            verbose=True,
            llm = get_openai_llm(temperature=0.3)
        )

    def process_query(self, query):
        relevant_chunks = self.vector_store.query(query)
        context = "\n".join(relevant_chunks)
        
        task = Task(
            description=f"""
            Based on the following context, answer the user's question:
            Question: {query}
            
            Context:
            {context}
            
            Provide a clear and concise answer based only on the information available in the context.
            """,
            agent=self.agent
        )
        
        return task

class ManagerAgent:
    def __init__(self):
        self.agent = Agent(
            role='Query Manager',
            goal='Route queries to appropriate PDF agents',
            backstory='I analyze queries and determine which PDF agent(s) should handle them',
            allow_delegation=True,
            verbose=True,
            llm = get_openai_llm(temperature=0.3)
        )

    def route_query(self, query, pdf_descriptions):
        task = Task(
            description=f"""
            Analyze the following query and determine which PDF(s) would be most relevant:
            Query: {query}
            
            Available PDFs:
            {pdf_descriptions}
            
            Return a list of PDF numbers (1, 2, or 3) that should handle this query.
            """,
            agent=self.agent
        )
        
        return task

class AggregatorAgent:
    def __init__(self):
        self.agent = Agent(
            role='Response Aggregator',
            goal='Combine multiple responses into a coherent answer',
            backstory='I synthesize information from multiple sources into clear, unified responses',
            allow_delegation=False,
            verbose=True,
            llm = get_openai_llm(temperature=0.3)
        )

    def aggregate_responses(self, responses):
        task = Task(
            description=f"""
            Combine the following responses into a single coherent answer:
            
            Responses:
            {responses}
            
            Provide a unified response that incorporates all relevant information without redundancy.
            """,
            agent=self.agent
        )
        
        return task

# Initialize agents
manager_agent = ManagerAgent()
pdf_agents = {
    i: PDFAgent(path, f"pdf_{i}_collection")
    for i, path in PDF_PATHS.items()
}
aggregator_agent = AggregatorAgent()

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(chat_query: ChatQuery):
    try:
        # Create routing task
        routing_task = manager_agent.route_query(chat_query.query, PDF_DESCRIPTIONS)
        
        # Create routing crew
        routing_crew = Crew(
            agents=[manager_agent.agent],
            tasks=[routing_task]
        )
        
        # Get routing result
        routing_result = routing_crew.kickoff()
        relevant_pdfs = [int(num) for num in routing_result.split() if num.isdigit() and int(num) in [1, 2, 3]]
        
        if not relevant_pdfs:
            return {"response": "I couldn't determine which PDF would be relevant for your query."}
        
        # Create tasks for PDF agents
        pdf_tasks = []
        for pdf_num in relevant_pdfs:
            task = pdf_agents[pdf_num].process_query(chat_query.query)
            pdf_tasks.append(task)
        
        # Create PDF processing crew
        pdf_crew = Crew(
            agents=[pdf_agents[i].agent for i in relevant_pdfs],
            tasks=pdf_tasks
        )
        
        # Get PDF responses
        pdf_responses = pdf_crew.kickoff()
        responses = [f"From PDF {num}: {resp}" for num, resp in zip(relevant_pdfs, pdf_responses)]
        
        # Create aggregation task
        aggregation_task = aggregator_agent.aggregate_responses(responses)
        
        # Create aggregation crew
        aggregation_crew = Crew(
            agents=[aggregator_agent.agent],
            tasks=[aggregation_task]
        )
        
        # Get final response
        final_response = aggregation_crew.kickoff()
        
        return {"response": final_response}
    
    except Exception as e:
        print("Exception--------------------------------------\n",e)
        raise HTTPException(status_code=500, detail=str(e))