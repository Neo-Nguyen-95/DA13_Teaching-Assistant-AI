#%% LIB
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import faiss

load_dotenv()

# OPEN AI API key
OPENAI_API_KEY = os.getenv("SECRETE_KEY")
# OPENAI_API_KEY = st.secrets["api"]["key"]

#%% GET KNOWLEDGE TEXT
def get_knowledge_text():
    with open("knowledge_base.txt", "r") as file:
        kb_content = file.read()
        
    with open("knowledge_qna.txt", "r") as file:
        qa_content = file.read()
        
    with open('knowledge_chatbot_info.txt', 'r') as file:
        info_content = file.read()
        
    # Split text into chunks
    knowledge_text = kb_content.split('\n')
    knowledge_text.extend(qa_content.replace('\n', '').split('---'))
    knowledge_text.append(info_content)
    
    return knowledge_text

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

 