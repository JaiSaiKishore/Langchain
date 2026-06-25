import os
from langchain_openai import ChatOpenAI   
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name():
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com"
    )

    name = llm.invoke("I have a dog and i want a cool name for it. Suggest me five cool names for my dog.")

    return name.content

print(generate_pet_name())
    