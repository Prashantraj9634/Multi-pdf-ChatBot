"""
app.py - Streamlit User Interface for the Multi-PDF AI Chatbot.
Provides controls for PDF upload, processing status,
interactive chat feed, conversational memory, and source citations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from rag import (
    load_and_split_pdfs,
    build_vector_store,
    get_existing_vector_store,
    ask_rag_pipeline,
)

# ==========================================
# 1. BULLETPROOF ENV LOADING
# ==========================================
# Force dotenv to look directly at the folder containing this app.py file
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

# override=True ensures newly saved keys overwrite any stale OS cache
load_dotenv(dotenv_path=ENV_PATH, override=True)
env_token = os.getenv("HF_TOKEN", "").strip()

PDF_UPLOAD_DIR = APP_DIR / "data" / "PDFs"
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Multi-PDF AI Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_existing_vector_store()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "hf_token" not in st.session_state:
    st.session_state.hf_token = env_token


# ==========================================
# 3. SIDEBAR: Upload & Configuration
# ==========================================
with st.sidebar:
    st.title("📚 Document Manager")
    st.caption("Upload PDFs and index them into ChromaDB")

    # API Token Detection & Fallback Input
    if st.session_state.hf_token:
        st.success("🔑 Hugging Face Token detected", icon="✅")
        # Allow user to change token if needed
        with st.expander("Change HF Token"):
            new_token = st.text_input("New Token", type="password", key="new_token_input")
            if st.button("Update Token"):
                if new_token.strip():
                    st.session_state.hf_token = new_token.strip()
                    st.rerun()
    else:
        st.error("⚠️ `HF_TOKEN` not found in `.env`")
        manual_token = st.text_input(
            "Paste HF Token here:",
            type="password",
            help="Get your free Read token at: https://huggingface.co/settings/tokens",
        )
        if manual_token.strip():
            st.session_state.hf_token = manual_token.strip()
            st.success("Token set for this session!")
            st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to index.",
    )

    if uploaded_files:
        st.write(f"📄 **Selected:** {len(uploaded_files)} file(s)")

    if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Extracting text, chunking, and creating embeddings..."):
                # Clean up existing files in the data directory
                for old_file in PDF_UPLOAD_DIR.iterdir():
                    if old_file.is_file() and old_file.name != ".gitkeep":
                        old_file.unlink()

                saved_paths = []
                for uploaded_file in uploaded_files:
                    target_path = PDF_UPLOAD_DIR / uploaded_file.name
                    with open(target_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(str(target_path))

                try:
                    # Ingestion Pipeline
                    chunks = load_and_split_pdfs(saved_paths)
                    vectorstore = build_vector_store(chunks)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.session_state.messages = []  # Reset conversation for new documents

                    st.success(
                        f"Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} file(s)!"
                    )
                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")

    st.divider()

    # Index Status Panel
    st.markdown("### System Status")
    if st.session_state.vectorstore is not None:
        st.success("Vector DB: **Active**")
        if st.session_state.processed_files:
            st.markdown("**Indexed Files:**")
            for fn in st.session_state.processed_files:
                st.caption(f"- {fn}")
    else:
        st.info("Vector DB: **Empty** (Upload PDFs to begin)")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 4. MAIN CHAT INTERFACE
# ==========================================
st.title("💬 Multi-PDF Conversational AI")
st.caption("Ask questions about your documents with exact page citations and context memory.")

# Render existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📍 View Source Citations"):
                for src in message["sources"]:
                    st.markdown(f"- **{src['source']}** — *Page {src['page']}*")

# Handle new user input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Guardrail 1: Check Token
    active_token = st.session_state.hf_token
    if not active_token:
        st.error("Missing Hugging Face Token. Please provide one in `.env` or in the sidebar.")
        st.stop()

    # Guardrail 2: Check Index State
    if st.session_state.vectorstore is None:
        st.warning("Please upload and click 'Process Documents' before asking questions.")
        st.stop()

    # Immediately render user question
    st.chat_message("user").markdown(prompt)

    # Format history turns for contextualization
    history_for_rag = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document chunks and generating answer..."):
            try:
                answer, sources = ask_rag_pipeline(
                    question=prompt,
                    chat_history=history_for_rag,
                    vectorstore=st.session_state.vectorstore,
                    hf_token=active_token,
                )

                st.markdown(answer)

                if sources:
                    with st.expander("📍 View Source Citations"):
                        for src in sources:
                            st.markdown(f"- **{src['source']}** — *Page {src['page']}*")

                # Persist turn to session state
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:
                st.error(f"Error during query execution: {str(e)}")