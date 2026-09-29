import sys
from pathlib import Path

# Ensure starter kit root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, List

from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    GROQ_MODELS,
    GEMINI_MODELS,
    KNOWLEDGE_DIR,
    CHROMA_DIR
)
from core.llm_client import call_llm
from core.rag import ingest_knowledge, retrieve, get_chroma_collection, clear_knowledge
from core.extraction import extract_structured_data
from core.safety_scaffold import scan_for_flags, semantic_check

st.set_page_config(
    page_title="GenAI Hackathon Starter Shell",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Sidebar: System Controls & Diagnostics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚡ System Diagnostics")
    st.caption("Domain-Agnostic GenAI Plumbing")
    
    # API Key Health Check
    groq_ready = bool(GROQ_API_KEY)
    gemini_ready = bool(GEMINI_API_KEY)
    
    st.markdown("**API Credentials:**")
    if groq_ready:
        st.success(f"Groq API: Connected", icon="✅")
    else:
        st.warning("Groq API: Missing / Placeholder in .env", icon="⚠️")
        
    if gemini_ready:
        st.success(f"Gemini API: Connected", icon="✅")
    else:
        st.warning("Gemini API: Missing / Placeholder in .env", icon="⚠️")
        
    if not groq_ready and not gemini_ready:
        st.error("No valid LLM API key detected! Set `GROQ_API_KEY` or `GEMINI_API_KEY` in `.env`.")

    st.divider()
    collection = get_chroma_collection()
    chunk_count = collection.count() if collection else 0
    st.metric("Vector DB Chunks", chunk_count)
    
    st.divider()
    st.subheader("Configured Models")
    st.write(f"**Groq Primary:** `{GROQ_MODELS[0]}`")
    st.write(f"**Gemini Fallback:** `{GEMINI_MODELS[0]}`")
    
    st.divider()
    col_ingest, col_clear = st.columns(2)
    with col_ingest:
        if st.button("🔄 Ingest / Sync", use_container_width=True):
            with st.spinner("Ingesting files from data/knowledge..."):
                count = ingest_knowledge(str(KNOWLEDGE_DIR))
                st.success(f"Ingested {count} chunks!")
                st.rerun()
    with col_clear:
        if st.button("🗑️ Rebuild", use_container_width=True):
            with st.spinner("Clearing & rebuilding knowledge base..."):
                count = ingest_knowledge(str(KNOWLEDGE_DIR), clear_existing=True)
                st.success(f"Rebuilt with {count} chunks!")
                st.rerun()

st.title("🚀 GenAI Hackathon Application")
st.caption("Multi-Provider Fallback | Local Chroma RAG | Multimodal Extraction | Two-Tier Guardrails")

tab_chat, tab_extract, tab_knowledge = st.tabs([
    "💬 AI Assistant (RAG)",
    "📷 Document / Image Extraction",
    "📚 Knowledge Management"
])

# ---------------------------------------------------------------------------
# TAB 1: RAG & Multi-Turn Chat Shell
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Interactive Knowledge Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_hdr, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "provider" in msg:
                st.caption(f"⚡ *Served by: {msg['provider']}*")
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Retrieved Knowledge Chunks"):
                    for idx, c in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {idx+1} ({c.get('source_file')}, distance: {c.get('distance')})**")
                        st.write(c.get("text"))

    # Chat Input
    user_query = st.chat_input("Enter your question or task...")
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 1. Tier 0 Deterministic Safety Scan
        t0_flag, t0_trigger = scan_for_flags(user_query)
        if t0_flag:
            flag_msg = f"⚠️ Request blocked by Tier 0 Security Filter: matched '{t0_trigger}'."
            st.session_state.messages.append({"role": "assistant", "content": flag_msg})
            with st.chat_message("assistant"):
                st.error(flag_msg)
        else:
            # 2. Vector Retrieval (Chroma)
            with st.spinner("Retrieving verified context from ChromaDB..."):
                chunks = retrieve(user_query, k=3, max_distance=0.70)
                
            context_text = "\n\n".join([f"[{c.source_file}]:\n{c.text}" for c in chunks]) if chunks else "No relevant context found."
            
            # 3. LLM Call
            with st.spinner("Synthesizing answer with multi-provider cascade..."):
                system_prompt = (
                    "You are a helpful, factual hackathon assistant. "
                    "Ground your answers strictly on the provided context if available.\n\n"
                    f"VERIFIED CONTEXT:\n{context_text}"
                )
                res = call_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_query,
                    temperature=0.1
                )

            with st.chat_message("assistant"):
                if res.success:
                    st.markdown(res.text)
                    st.caption(f"⚡ *Served by: {res.provider_used}*")
                    sources_data = [{"source_file": c.source_file, "distance": c.distance, "text": c.text} for c in chunks]
                    if chunks:
                        with st.expander("📚 Retrieved Knowledge Chunks"):
                            for idx, c in enumerate(chunks):
                                st.markdown(f"**Chunk {idx+1} ({c.source_file}, distance: {c.distance})**")
                                st.write(c.text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res.text,
                        "provider": res.provider_used,
                        "sources": sources_data
                    })
                else:
                    err_msg = f"⚠️ Service Failure: {res.error}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

# ---------------------------------------------------------------------------
# TAB 2: Structured Extraction (Text or Image)
# ---------------------------------------------------------------------------
with tab_extract:
    st.subheader("Multimodal Structured Data Extraction")
    st.caption("Passes input directly to Pydantic schema validation with automatic feedback retry.")

    # Generic Starter Schema - EDIT/REPLACE ONCE PROBLEM STATEMENT IS KNOWN
    class GenericItemSchema(BaseModel):
        title: str = Field(description="Title or main entity name")
        category: str = Field(description="Category or classification")
        summary: str = Field(description="2-sentence summary of the content")
        tags: List[str] = Field(default_factory=list, description="Relevant keyword tags")

    extract_mode = st.radio("Input Type", ["Text", "Image"], horizontal=True)

    if extract_mode == "Text":
        raw_input = st.text_area("Paste unstructured text to parse:", height=150)
        if st.button("Extract Data from Text", type="primary"):
            if not raw_input.strip():
                st.warning("Please enter text to extract.")
            else:
                with st.spinner("Extracting with schema validation..."):
                    result = extract_structured_data(schema=GenericItemSchema, text=raw_input)
                    if result.success and result.validated:
                        st.success("Extraction Succeeded!")
                        st.json(result.validated.model_dump())
                    else:
                        st.error(f"Extraction Failed: {result.error}")
                        if result.raw_text:
                            st.text_area("Raw LLM Text", result.raw_text)
    else:
        uploaded_img = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            st.image(uploaded_img, width=300)
            if st.button("Extract Data from Image", type="primary"):
                with st.spinner("Processing multimodal image extraction..."):
                    img_bytes = uploaded_img.getvalue()
                    mime_type = uploaded_img.type or "image/png"
                    result = extract_structured_data(
                        schema=GenericItemSchema,
                        image_bytes=img_bytes,
                        mime_type=mime_type
                    )
                    if result.success and result.validated:
                        st.success("Multimodal Extraction Succeeded!")
                        st.json(result.validated.model_dump())
                    else:
                        st.error(f"Extraction Failed: {result.error}")
                        if result.raw_text:
                            st.text_area("Raw LLM Text", result.raw_text)

# ---------------------------------------------------------------------------
# TAB 3: Knowledge Management
# ---------------------------------------------------------------------------
with tab_knowledge:
    st.subheader("Local Knowledge Store")
    st.markdown(f"Knowledge files are loaded from: `{KNOWLEDGE_DIR}`")
    st.markdown("""
    **How to add domain knowledge on the day:**
    1. Drop `.md` or `.txt` guidelines/policies/manuals into `data/knowledge/`
    2. Click **'🔄 Ingest / Sync'** or **'🗑️ Rebuild'** in the sidebar.
    3. All chunks are immediately indexed and searchable via ChromaDB!
    """)
