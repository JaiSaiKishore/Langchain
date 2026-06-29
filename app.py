import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0.7,
    base_url = "https://models.inference.ai.azure.com"
)

response = llm.invoke("Explain langchain in simple words.")

print(response)