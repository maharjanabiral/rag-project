import streamlit as st
import asyncio
import os
from pathlib import Path
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Page Config
st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("📄 AI Document Assistant")

# Define paths
SOURCE_DIR = "./source"
DB_DIR = "./db/chroma"

# --- Cache the Pipeline ---
# This prevents reloading the heavy embedding model on every user interaction
@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline(source_dir=SOURCE_DIR, persist_dir=DB_DIR)
    pipeline.initialize()
    return pipeline

pipeline = get_pipeline()

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Document Manager")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        # Save file to disk so PyMuPDF can read it
        save_path = Path(SOURCE_DIR) / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            pipeline.process_file(str(save_path))
        st.success("File processed & added to knowledge base!")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area
if prompt := st.chat_input("Ask a question about your documents..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run async function in sync Streamlit environment
            response = asyncio.run(pipeline.query(prompt))
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})