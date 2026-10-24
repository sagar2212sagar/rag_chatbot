import html
import os

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI

VS_ENDPOINT = os.getenv("VS_ENDPOINT", "rag_vs_endpoint")
VS_INDEX = os.getenv("VS_INDEX", "rag_chatbot_dev.bronze.rag_chatbot_index")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
RESULT_COLUMNS = ["chunk_id", "file_name", "page_number", "chunk_text"]

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the document "
    "excerpts provided. If the answer is not in the excerpts, say you don't know. "
    "Cite your sources inline as [file_name, p.page_number]."
)

st.set_page_config(page_title="Document Assistant", page_icon="📄", layout="wide")


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Page background and content width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 960px;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
}
.hero h1 {
    margin: 0 0 6px 0;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero p {
    margin: 0;
    opacity: 0.92;
    font-size: 0.95rem;
}
.hero .index-tag {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.18);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    background: rgba(128, 128, 128, 0.04);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessageAvatarUser"] + div {
    background: rgba(99, 102, 241, 0.06);
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}
[data-testid="stChatInput"] textarea {
    font-size: 0.95rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.12);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 1.05rem;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #8b5cf6;
    color: #8b5cf6;
    transform: translateY(-1px);
}

/* Expanders (sources) */
[data-testid="stExpander"] {
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.15);
    overflow: hidden;
}

/* Source cards */
.source-card {
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    background: rgba(128, 128, 128, 0.06);
    border-left: 3px solid #8b5cf6;
}
.source-card .source-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 6px;
}
.source-card .score {
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(139, 92, 246, 0.15);
    color: #8b5cf6;
    font-weight: 600;
}
.source-card .source-meta {
    font-size: 0.78rem;
    opacity: 0.7;
    margin-bottom: 6px;
}
.source-card .source-body {
    font-size: 0.85rem;
    line-height: 1.5;
    opacity: 0.9;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    opacity: 0.75;
}
.empty-state .icon {
    font-size: 2.4rem;
    margin-bottom: 8px;
}
</style>
"""


def apply_styles():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        f"""
        <div class="hero">
            <h1>📄 Document Assistant</h1>
            <p>Ask anything about your documents. Answers come with citations to the source pages.</p>
            <span class="index-tag">{html.escape(VS_INDEX)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _workspace_host() -> str:
    host = WorkspaceClient().config.host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "https://" + host
    return host


def _vector_client() -> VectorSearchClient:
    cfg = WorkspaceClient().config
    host = _workspace_host()

    if cfg.client_id and cfg.client_secret:
        return VectorSearchClient(
            workspace_url=host,
            service_principal_client_id=cfg.client_id,
            service_principal_client_secret=cfg.client_secret,
            disable_notice=True,
        )
    if cfg.token:
        return VectorSearchClient(
            workspace_url=host,
            personal_access_token=cfg.token,
            disable_notice=True,
        )
    raise RuntimeError(
        "No Databricks credentials found. Set up a CLI profile or run inside a Databricks App."
    )


def _llm_client() -> OpenAI:
    """OpenAI-compatible client pointed at the workspace's serving endpoints.

    Built per request so OAuth tokens (used by app service principals) stay fresh.
    """
    w = WorkspaceClient()
    token = w.config.authenticate()["Authorization"].split(" ", 1)[1]
    return OpenAI(api_key=token, base_url=f"{_workspace_host()}/serving-endpoints")


@st.cache_resource
def get_index():
    return _vector_client().get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)


def retrieve(index, question: str, k: int) -> list[dict]:
    """Return the top-k chunks for a question as dictionaries."""
    res = index.similarity_search(
        query_text=question,
        columns=RESULT_COLUMNS,
        num_results=k,
    )
    rows = res.get("result", {}).get("data_array", [])
    # Results come back in the requested column order, with score appended last
    return [dict(zip(RESULT_COLUMNS + ["score"], row)) for row in rows]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['file_name']} p.{c['page_number']}]\n{c['chunk_text']}" for c in chunks
    )


def stream_answer(question: str, chunks: list[dict], history: list[dict]):
    """Yield answer tokens from the LLM as they arrive."""
    context = build_context(chunks) or "No relevant excerpts were found."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # last three exchanges for follow-up questions
    messages.append({
        "role": "user",
        "content": f"Excerpts:\n{context}\n\nQuestion: {question}",
    })

    stream = _llm_client().chat.completions.create(
        model=LLM_ENDPOINT,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def render_sources(sources: list[dict]):
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for s in sources:
            preview = s["chunk_text"][:400]
            if len(s["chunk_text"]) > 400:
                preview += "..."
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">
                        <span>{html.escape(str(s['file_name']))}</span>
                        <span class="score">{s['score']:.3f}</span>
                    </div>
                    <div class="source-meta">Page {html.escape(str(s['page_number']))}</div>
                    <div class="source-body">{html.escape(preview)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------- UI ----------

apply_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)
    show_sources = st.toggle("Show sources", value=True)
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

render_hero()

try:
    index = get_index()
except Exception as e:
    st.error(f"Could not connect to Vector Search: {e}")
    st.stop()

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">💬</div>
            <div>Start by asking a question about your documents.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    avatar = "🧑" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        if show_sources and m.get("sources"):
            render_sources(m["sources"])

if question := st.chat_input("Ask a question about your documents..."):
    # History sent to the model contains only role and content, not the sources
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Searching documents..."):
                chunks = retrieve(index, question, k)
            answer = st.write_stream(stream_answer(question, chunks, history))
            if show_sources:
                render_sources(chunks)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": chunks,
            })
        except Exception as e:
            st.error(f"Something went wrong: {e}")