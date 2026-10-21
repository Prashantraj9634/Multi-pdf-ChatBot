"""
rag.py - Robust Retrieval-Augmented Generation (RAG) module.
Features:
- Dual-engine PDF extraction (pdfplumber + PyPDFLoader fallback)
- Safe non-empty chunk validation
- Local Sentence Transformers embeddings (all-MiniLM-L6-v2)
- Local ChromaDB vector store
- Featherless/Hugging Face conversational API support with fallbacks
- Grounded answer generation with source citations
"""

import os
import shutil
from typing import List, Tuple, Dict, Any
from pathlib import Path

import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from huggingface_hub import InferenceClient

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Initializes local Sentence Transformers embeddings running on CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_text_from_single_pdf(file_path: str) -> List[Document]:
    """Extracts text using pdfplumber first, falling back to PyPDFLoader."""
    filename = Path(file_path).name
    docs: List[Document] = []

    # Attempt 1: pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text(layout=False) or ""
                if not text.strip():
                    text = page.extract_text(layout=True) or ""

                clean_text = text.strip()
                if clean_text:
                    docs.append(
                        Document(
                            page_content=clean_text,
                            metadata={"source": filename, "page": page_idx + 1},
                        )
                    )
    except Exception as e:
        print(f"[pdfplumber] Note for {filename}: {e}")

    # Attempt 2: PyPDFLoader fallback
    if not docs:
        try:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            for doc in raw_docs:
                clean_text = doc.page_content.strip()
                if clean_text:
                    page_num = doc.metadata.get("page", 0)
                    docs.append(
                        Document(
                            page_content=clean_text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1 if isinstance(page_num, int) else 1,
                            },
                        )
                    )
        except Exception as e:
            print(f"[PyPDFLoader] Note for {filename}: {e}")

    return docs


def load_and_split_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Extracts documents and creates non-empty chunks."""
    raw_documents: List[Document] = []

    for path in pdf_paths:
        file_docs = extract_text_from_single_pdf(path)
        raw_documents.extend(file_docs)

    if not raw_documents:
        raise ValueError(
            "No extractable text was found in the uploaded PDF(s). "
            "Please ensure the PDF contains selectable text."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = text_splitter.split_documents(raw_documents)

    valid_chunks = [
        chunk for chunk in all_chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

    if not valid_chunks:
        raise ValueError("Document chunking produced 0 valid text passages.")

    return valid_chunks


def build_vector_store(chunks: List[Document]) -> Chroma:
    """Creates and persists a fresh ChromaDB collection."""
    if not chunks:
        raise ValueError("Cannot initialize ChromaDB with zero chunks.")

    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception:
            pass

    embeddings = get_embedding_function()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def get_existing_vector_store() -> Chroma | None:
    """Loads an existing Chroma vectorstore from disk if present."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        try:
            embeddings = get_embedding_function()
            return Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
        except Exception:
            return None
    return None


def call_huggingface_llm(user_prompt: str, system_prompt: str, hf_token: str) -> str:
    """
    Calls Hugging Face via the exact conversational/chat endpoint
    supported by Featherless AI and Hugging Face serverless.
    """
    client = InferenceClient(api_key=hf_token)

    candidate_models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    last_error = None

    for model in candidate_models:
        # Method 1: Conversational API (Supported by featherless-ai)
        try:
            conv_response = client.conversational(
                model=model,
                text=user_prompt,
                parameters={"temperature": 0.1, "max_length_seconds": 15},
            )
            generated_text = conv_response.get("generated_text", "").strip()
            if generated_text:
                return generated_text
        except Exception as e:
            last_error = e

        # Method 2: Standard chat.completions fallback
        try:
            chat_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.1,
            )
            content = chat_resp.choices[0].message.content.strip()
            if content:
                return content
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Hugging Face Inference call failed: {str(last_error)}")


def contextualize_question(
    question: str,
    chat_history: List[Dict[str, str]],
    hf_token: str,
) -> str:
    """Reformulates follow-up queries using prior chat history."""
    if not chat_history:
        return question

    history_lines = []
    for turn in chat_history[-4:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {turn['content']}")
    history_str = "\n".join(history_lines)

    user_query = f"Chat History:\n{history_str}\n\nFollow-up question: {question}\nRephrase into one single standalone search query:"
    system_instruction = "You are a concise query reformulator. Return ONLY the standalone question."

    try:
        standalone_q = call_huggingface_llm(user_query, system_instruction, hf_token)
        clean_q = standalone_q.split("\n")[0].replace("Standalone query:", "").replace("Standalone Question:", "").strip()
        return clean_q if clean_q else question
    except Exception:
        return question


def ask_rag_pipeline(
    question: str,
    chat_history: List[Dict[str, str]],
    vectorstore: Chroma,
    hf_token: str,
    top_k: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Executes the complete conversational RAG pipeline."""
    # 1. Standalone search query
    standalone_query = contextualize_question(question, chat_history, hf_token)

    # 2. Vector search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    retrieved_docs: List[Document] = retriever.invoke(standalone_query)

    if not retrieved_docs:
        return (
            "I could not find any relevant information in the uploaded documents to answer your question.",
            [],
        )

    # 3. Context packaging
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"[Passage {i} | Source: {doc.metadata.get('source', 'Document')} | Page: {doc.metadata.get('page', 1)}]\n{doc.page_content}"
        )
    combined_context = "\n\n".join(context_blocks)

    # 4. Strict Grounded QA Prompt
    system_prompt = (
        "You are an accurate, factual document assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer cannot be determined strictly from the context, state: "
        "'I could not find the answer in the provided documents.' "
        "Do not make assumptions or extrapolate."
    )
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {question}\n\nAnswer:"

    answer = call_huggingface_llm(user_prompt, system_prompt, hf_token)

    # 5. Extract citations
    sources: List[Dict[str, Any]] = []
    seen = set()
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": src, "page": page})

    return answer, sources