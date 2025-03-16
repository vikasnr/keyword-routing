from typing import List

class AggregatorAgent:
    """Class-based implementation of the aggregator agent."""
    
    def __init__(self, llm):
        """Initialize with an LLM instance."""
        self.llm = llm
    
    def aggregate(self, responses: List[str]):
        """Aggregates the retrieved texts and passes them to LLM."""
        message = [
            ("system", "You are an aggregator assistant. You will combine the responses from multiple agents."),
            ("user", f"""
            Aggregate the response and make it look good using markdown. 
            Please format the response using **Markdown**. Follow these guidelines:  
                - Use **bold** and *italic* for emphasis.  
                - Use bullet points (•) or numbered lists when necessary.  
                - Format code snippets inside triple backticks (` ``` `).  
                - Add appropriate **headings** (e.g., `## Summary`, `### Key Points`).  
                - Use emojis where relevant to enhance readability. 
             Only include the aggregated responses in the response. Do not include any other text in the response.\n\n    
                    Responses: {responses}\n """)
        ]
        
        res = self.llm.invoke(message)
        return res.content