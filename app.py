import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.ingestor import ingest, load_vectorstore, CHROMA_PERSIST_DIR
from rag.retriever import build_retriever
from rag.chain import build_chain
from utils.helpers import format_source

st.set_page_config(page_title="Chat with your PDFs", page_icon="📄", layout="wide")

# ── session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "chain": None,
    "ingested_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── sidebar: upload + ingest ──────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Chat with PDFs")
    st.markdown("Upload one or more PDFs, then ask anything about them.")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    ingest_btn = st.button("Ingest / Reload", use_container_width=True, type="primary")

    if ingest_btn and uploaded:
        with st.spinner("Extracting, chunking, embedding…"):
            # write uploads to a temp dir
            tmp_dir = tempfile.mkdtemp()
            paths = []
            for uf in uploaded:
                dest = os.path.join(tmp_dir, uf.name)
                with open(dest, "wb") as f:
                    f.write(uf.read())
                paths.append(dest)

            # wipe old chroma store so re-uploads don't duplicate
            if os.path.exists(CHROMA_PERSIST_DIR):
                shutil.rmtree(CHROMA_PERSIST_DIR)

            vs = ingest(paths)
            retriever = build_retriever(vs)
            st.session_state.chain = build_chain(retriever)
            st.session_state.ingested_files = [uf.name for uf in uploaded]
            st.session_state.messages = []

            shutil.rmtree(tmp_dir)

        st.success(f"Ingested {len(paths)} file(s).")

    if st.session_state.ingested_files:
        st.markdown("**Loaded files:**")
        for name in st.session_state.ingested_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chain:
            st.session_state.chain.memory.clear()
        st.rerun()


# ── main area: chat ───────────────────────────────────────────────────────────
st.header("Ask your documents")

if not st.session_state.chain:
    st.info("Upload PDFs and click **Ingest / Reload** to get started.")
    st.stop()

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# new question
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        source_docs = []

        # streaming via callbacks
        with st.spinner(""):
            result = st.session_state.chain.invoke({"question": prompt})
            full_answer = result["answer"]
            source_docs = result.get("source_documents", [])

        placeholder.markdown(full_answer)

        # deduplicate sources
        seen = set()
        sources = []
        for doc in source_docs:
            label = format_source(doc)
            if label not in seen:
                seen.add(label)
                sources.append(label)

        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
