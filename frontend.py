import streamlit as st
import requests

st.set_page_config(page_title="RAG Bot")
st.title("Research Paper Companion : RAG Approach")

# Sidebar for Upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a file", type="pdf")
    if uploaded_file and st.button("Build Knowledge Base"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        res = requests.post("http://localhost:8000/upload", files=files)
        st.success(res.json()["message"])

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post("http://localhost:8000/query", json={"question": prompt})
            answer = res.json()["answer"]
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})