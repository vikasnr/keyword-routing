import json
import re
import os
import ast
class KeywordDictionary:
    """Handles retrieval of keyword dictionaries from a JSON file."""
    
    def __init__(self, file_path: str = "keywords.json"):
        """Initialize with a file path to the JSON dictionary."""
        self.file_path = file_path

    def extract_keywords(self,llm,collection_name, combined_chunks:str):
        prompt = f"""Extract only the keywords from the following text. Return response in a python list. Do not include any other text in the response:\n\n
                {combined_chunks}\n\n
                example response:
                ['keyword1', 'keyword2', 'keyword3']
                """
        # print(prompt)
        

        message = [
            (
                "system",
                "You are a helpful assistant who extracts keywords from the text."
            ),
                ("user", prompt)
            ]
        response = llm.invoke(message)
        response = response.content
        
        match = re.search(r"\[.*\]", response, re.DOTALL)

        if match:
            keyword_list = ast.literal_eval(match.group(0))  # Safely convert to Python list
            print(keyword_list)
        else:
            keyword_list = []
            print("No list found in response.")
        # Save keywords to a dictionary
        keywords_dict = {}
        keywords_dict[collection_name] = keyword_list

        # Open the file and update the dictionary
        keywords_file = "keywords.json"
        if os.path.exists(keywords_file):
            with open(keywords_file, "r") as f:
                existing_keywords = json.load(f)
        else:
            existing_keywords = {}

        existing_keywords.update(keywords_dict)

        with open(keywords_file, "w") as f:
            json.dump(existing_keywords, f, indent=4)
    
    def get_keywords(self:tuple):
        """Retrieves keyword dictionaries from the JSON file."""
        with open(self.file_path, "r") as file:
            keywords_data = json.load(file)
        
        return (
            keywords_data['keywords']["insurance"],
            keywords_data['keywords']["rivers"],
            keywords_data['keywords']["wildlife"]
        )