#%% LIB
from openai import OpenAI
import numpy as np
import faiss


# FOR OFFLINE MODEL
# from dotenv import load_dotenv
# import os
# load_dotenv()
# OPENAI_API_KEY = os.getenv("SECRETE_KEY")

# FOR ONLINE MODEL
import streamlit as st
OPENAI_API_KEY = st.secrets["api"]["key"]

#%% GET KNOWLEDGE TEXT
def get_clean_text(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    
    return (content.replace("*", "").replace("#", "").replace("\n", "")
            .replace("  ", " ")
            .strip()
            .split('---')
            )

# get_clean_text("knowledge_base.txt")  # Test function

def get_knowledge_text():
    file_list = ["knowledge_base.txt", 
                 "knowledge_qna.txt", 
                 "knowledge_chatbot_info.txt"]
    
    result = []
    for file_path in file_list:
        result.extend(get_clean_text(file_path))
    
    return result
    
# get_knowledge_text()  # Test function
   

#%% EMBEDDING
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        input = text,
        model='text-embedding-ada-002'
        )
    
    embedding = response.data[0].embedding
    
    return np.array(embedding)


#%% USE EMBEDDED DATA

def retrieve_relevant_knowledge(knowledge_text, 
                                knowledge_embedded_file, 
                                query, 
                                top_k=3):
    query_embedding = get_embedding(query).reshape(1, -1)
    index = faiss.read_index(knowledge_embedded_file)
    
    distances, indices = index.search(query_embedding, 3)
    
    text_list = [knowledge_text[i] for i in indices[0]]
    
    return ' '.join(text_list)

 