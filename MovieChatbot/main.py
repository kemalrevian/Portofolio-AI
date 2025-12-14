import streamlit as st
from agent import build_agent

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="🎬 Movie AI Chatbot",
    page_icon="🎥"
)

st.title("🎬 Movie Chatbot")

# ======================
# Init Agent (ONCE)
# ======================
@st.cache_resource
def load_agent():
    return build_agent()

agent = load_agent()

# ======================
# Chat Session
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# Input
# ======================
if prompt := st.chat_input("Tanya rekomendasi film..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke({
                "messages": st.session_state.messages
            })
            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })