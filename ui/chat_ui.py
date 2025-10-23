import streamlit as st
import requests
import os

# FastAPI backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="MemoraAI Chat", page_icon="🧠", layout="centered")

st.title("🧠 MemoraAI — Persistent Memory Chat")

# Sidebar for project selection
project_id = st.sidebar.text_input("Project ID", value="default_project")
st.sidebar.info("All context is grouped by project ID.")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("Ask something..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to FastAPI backend
    with st.spinner("Thinking..."):
        try:
            res = requests.post(
                f"{BACKEND_URL}/ask",
                json={"project_id": project_id, "query": prompt},
                timeout=60,
            )
            data = res.json()
            response = data.get("response", "No response received.")
        except Exception as e:
            response = f" Error: {e}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Button to reset chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

