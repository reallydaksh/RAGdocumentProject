"""AskMyDocs — chat with your documents using Retrieval-Augmented Generation.

Run:  streamlit run app.py
"""

import hashlib

import streamlit as st

from rag import RAGPipeline, Settings

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")


def files_fingerprint(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.name):
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()


def init_state() -> None:
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("fingerprint", None)
    st.session_state.setdefault("messages", [])


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            match = []
            if s["dense_rank"]:
                match.append(f"semantic #{s['dense_rank']}")
            if s["keyword_rank"]:
                match.append(f"keyword #{s['keyword_rank']}")
            st.markdown(f"**[{s['n']}] {s['source']} — page {s['page']}**  \n<small>matched by: {', '.join(match)}</small>", unsafe_allow_html=True)
            st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))


init_state()
settings = Settings()

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📄 AskMyDocs")
    st.caption("Upload documents, then ask questions. Answers are grounded in your files and cited by page.")

    try:
        settings.validate()
        st.success(f"Provider: **{settings.provider}**", icon="✅")
    except ValueError as e:
        st.error(str(e), icon="🔑")
        st.stop()

    uploaded = st.file_uploader("Upload PDF, TXT or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    settings.top_k = st.slider("Excerpts retrieved per question (top-k)", 2, 10, settings.top_k)

    if uploaded:
        fp = files_fingerprint(uploaded)
        if fp != st.session_state.fingerprint:
            with st.spinner("Reading, chunking and embedding your documents…"):
                try:
                    pipeline = RAGPipeline(settings)
                    pipeline.index_files([(f.name, f.getvalue()) for f in uploaded])
                except Exception as e:  # surface API / parsing errors in the UI
                    st.error(f"Indexing failed: {e}")
                    st.stop()
            st.session_state.pipeline = pipeline
            st.session_state.fingerprint = fp
            st.session_state.messages = []
        stats = st.session_state.pipeline.stats
        st.info(f"Indexed **{stats['files']}** file(s) · **{stats['pages']}** pages · **{stats['chunks']}** chunks")
    else:
        st.session_state.pipeline = None
        st.session_state.fingerprint = None

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Pipeline: PDF parsing → sentence-aware chunking → embeddings → FAISS + BM25 hybrid search (RRF) → LLM answer with citations")

# ---------------------------------------------------------------- Main chat
st.header("Chat with your documents")

pipeline: RAGPipeline | None = st.session_state.pipeline
if pipeline is None:
    st.markdown(
        "👈 **Upload one or more documents to get started.**\n\n"
        "Try a policy handbook, research paper, contract or user manual — then ask things like:\n"
        "- *What is the notice period for resignation?*\n"
        "- *Summarise the key findings in section 3.*\n"
        "- *Which clauses mention penalties?*\n\n"
        "A sample document is included in `sample_docs/`."
    )
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                results = pipeline.retrieve(question, settings.top_k)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            answer = st.write_stream(pipeline.answer_stream(question, results, history))
        except Exception as e:
            answer = f"⚠️ Something went wrong: {e}"
            st.error(answer)
            results = []

        sources = [
            {
                "n": n,
                "source": r.chunk.source,
                "page": r.chunk.page,
                "text": r.chunk.text,
                "dense_rank": r.dense_rank,
                "keyword_rank": r.keyword_rank,
            }
            for n, r in enumerate(results, start=1)
        ]
        if sources:
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
