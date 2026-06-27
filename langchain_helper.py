import os
from langchain_openai import ChatOpenAI   
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url="https://models.inference.ai.azure.com"
    )
    prompt_template_name = PromptTemplate(
        input_variables = ["animal_type", "pet_color"],
        template = f"I have a {animal_type} and is it {pet_color} in colort and i want a cool name for it. Suggest me five cool names for my {animal_type}."
    )

    name_chain = LLMChain(
        llm = llm,
        prompt = prompt_template_name,
        output_key = "pet_name"
    )

    response = name_chain({"animal_type":animal_type, "pet_color": pet_color})

    return response



    

# print(generate_pet_name("cat", "grey and black spotted"))

    