import streamlit as st
from agent import iternary_agent

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="✈️ YukJalan.ai",
    page_icon="🌍"
)

st.title("✈️ YukJalan.ai")
st.subheader("Simple Itinerary Chatbot")

# ======================
# Init Agent
# ======================
@st.cache_resource
def load_agent():
    return iternary_agent()

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
if prompt := st.chat_input("Mau healing ke mana hari ini?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = agent.invoke({
                "messages": st.session_state.messages
            })
            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
