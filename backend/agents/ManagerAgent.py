import json
from langchain_core.prompts import ChatPromptTemplate

class ManagerAgent:
    """Determines which PDF agent(s) should process a query."""
    
    def __init__(self, llm, keywords: tuple):
        """Initialize with an LLM instance and keyword dictionary."""
        self.llm = llm
        self.keyword_dict = keywords
    
    def route(self, question: str):
        """Routes the query to the appropriate data sources."""
        insurance_keys, rivers_keys, wildlife_keys = self.keyword_dict
        
        system_prompt = f"""You are an expert at routing a user question to the appropriate data source(s).
        
        Based on the user's question, you need to route the question to the appropriate data source(s). The data sources are:
        ["insurance"] for insurance-related queries
            if the question contains any of the following keywords: {insurance_keys}
        ["rivers"] for rivers-related queries
            if the question contains any of the following keywords: {rivers_keys}
        ["wildlife"] for wildlife-related queries
            if the question contains any of the following keywords: {wildlife_keys}
        If there are multiple data sources, you should respond.
            ["insurance", "rivers"]

        If the question does not contain any of the keywords, you should not route the question to any data source.
        Respond:
            ["insurance", "rivers", "wildlife"]

        Don't mention any other text in response except the list of data sources.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        router = prompt | self.llm
        result = router.invoke({"question": question})
        return json.loads(result.content)
