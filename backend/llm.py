from langchain_mistralai import ChatMistralAI
import os

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = ""


def get_llm():
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_retries=2,
    )
