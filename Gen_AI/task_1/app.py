from openai import OpenAI
import streamlit as st

st.title("Gen AI : A Code Correction using AI")

prompt = st.text_input("User input")

f = open("C:\\Users\\Hello\\Desktop\\Gen_AI\\api_key\\.key.txt")

OPENAI_API_KEY = f.read()

client = OpenAI(api_key = OPENAI_API_KEY)

def chat_bot(prompt):
    response = client.chat.completions.create(
                      model="gpt-3.5-turbo",
                      messages=[
                        {"role": "system", "content": """You are a teaching assistant for students .. 
                         you are need to explain and fix the bugs and explain student where he is done mistakes
                                        """},
                        {"role": "user", "content": prompt}
                      ]
                )
    return response.choices[0].message.content
if st.button("Code Fixer"):

    st.write(chat_bot(prompt))