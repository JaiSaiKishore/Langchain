import streamlit as st
from langchain_helper import generate_pet_name
st.title("Pet Name Generator")

animal_type = st.sidebar.selectbox("what is your pert?", ("Dog","Cat", "Cow", "Bird", "Hamster"))

pet_color = st.sidebar.text_area(label = f"what is the color of your {animal_type}")

if(pet_color):
    response = generate_pet_name(animal_type, pet_color)
    st.text(response["pet_name"])