#%% LIBRARY
from openai import OpenAI
import streamlit as st
from business import get_knowledge_text, retrieve_relevant_knowledge

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("SECRETE_KEY")

#%% SIDER BAR INFOR & SIGN IN
st.set_page_config(
    page_title = 'Physics Assistant',
    page_icon = '🔧'
    )

with st.sidebar:
    st.image('teacher_image.png')
    st.markdown("""
                **GIỚI THIỆU**
                
                Xin chào các em đã đến với trợ lí ảo học tập của thầy, Neon. 
                Neon sử dụng công nghệ trí tuệ nhân tạo, có thể giúp
                các em giải đáp thắc mắc trong giờ học của thầy.
                
                
                **ĐĂNG NHẬP**
                
                """)
      
    # To be delete later
    openai_api_key = st.text_input("OpenAI API Key",
                                   key="API key",
                                   type="password")
    
    # openai_api_key = st.secrets["api"]["key"]
    
    passcode = st.text_input("Nhập ngày sinh nhật của bạn",
                             type="password")
    
    # passcode_system = st.secrets["passcode"]["key"]
    # passcode_system = "1234"
    
#%% INPUT FOR AI
knowledge_text = get_knowledge_text()

sys_msg = """
Bạn là một giáo viên vật lí, bạn chỉ trả lời câu hỏi thông qua kiến thức vật lí hoặc 
thông tin tham khảo thêm từ người dùng. 
Nếu câu hỏi không trong sáng, từ chối trả lời một cách lễ phép.
Sử dụng ngôn ngữ trong sáng. Nếu có công thức toán học, viết ở giữa hai dấu $.
Trả lời ngắn gọn trong khoảng ít hơn 300 từ.
Nếu có thông tin về nguồn trích dẫn, nêu lên trong câu trả lời.
Trả lời đơn giản, đối tượng là học sinh trung học cơ sở.
"""
        
#%% MAIN SECTION
st.title("Neon AI")
st.caption("Trợ giảng Khoa học tự nhiên từ Trí tuệ nhân tạo")

if "messages" not in st.session_state:
    # Initial key-value in session state
    st.session_state["messages"] = [
        {"role": "system",
         "content": sys_msg},
        
        # Welcome message
        {"role": "assistant", 
         "content": 
             """Xin chào, mình là trợ lí ảo của thầy Dũng, giáo viên tại 
             STEM SPACE.
             Mình có thể giúp gì cho bạn?
             """
         }
        ]
    
# Show conversation: role-by-role
for msg in st.session_state.messages:
    if msg["role"] in ["assistant", "user"]:
        st.chat_message(msg["role"]).write(msg["content"])
    

    
if prompt := st.chat_input():  # Chat box

    # if not openai_api_key:
    #     st.info("Please add your key")
    #     st.stop()
        
    if len(passcode) == 0:
        st.info("Code không hợp lệ, xin thử lại!")
        st.stop()
        
    # Show what the user types
    st.chat_message("user").write(prompt)
    
    # Retrieve relevant knowledge from prompt
    relevant_knowledge = retrieve_relevant_knowledge(
        knowledge_text,
        "knowledge_index.faiss",
        prompt,
        top_k=3
        )


    st.session_state.messages.append(
        {"role": "system", "content": f"""
        Tham khảo thêm kiến thức từ {relevant_knowledge}
        """}
        )
    
    
    # Add user's promp to message history (stored in session state)    
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
        )

    client = OpenAI(api_key=openai_api_key)
    
    # st.session_state.messages
    
    # Get response from API & store it in session state
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = st.session_state.messages  # Input full chat history
        )
    
    msg = response.choices[0].message.content  # Current response
    
    st.session_state.messages.append(
        {"role": "assistant", "content": msg}
        )
    
    # Show reponse
    with st.chat_message("assistant"):
        st.markdown(msg.replace('\[', '$').replace('\]','$'))
    