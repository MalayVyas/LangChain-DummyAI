# -------------------------------
# LangChain × Gemini RAG (no-Vertex) with Local Embedding Fallback
# -------------------------------
import os
import io
import re
import json
import time
import asyncio
from typing import List, Dict, Any, Tuple

# Silence gRPC noise from google SDK
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# LangChain bits
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# PDF parsing
from pypdf import PdfReader

# Google Gemini SDK (AI Studio; NOT Vertex)
import google.generativeai as genai


# =========================
# Load .env and configure Gemini
# =========================
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=dotenv_path, override=True)

def _safe_secret(k: str) -> str:
    try:
        return st.secrets.get(k)  # only if secrets.toml exists
    except Exception:
        return ""

def get_api_key() -> str:
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or _safe_secret("GOOGLE_API_KEY")
        or _safe_secret("GEMINI_API_KEY")
        or ""
    )

API_KEY = get_api_key()
if API_KEY:
    genai.configure(api_key=API_KEY)

# =========================
# UI
# =========================
st.set_page_config(page_title="LangChain × Gemini — Multi-PDF QA (Local Embeds Option)", page_icon="📚", layout="wide")
st.title("📚 LangChain × Gemini — Ask your PDFs (with Local Embeddings)")

# Helpers for retry/fallback
RE_SECONDS = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.I)
def _parse_retry_seconds(err_msg: str, default_sec: float = 2.0) -> float:
    m = RE_SECONDS.search(err_msg or "")
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return default_sec

def _is_quota_or_rate(msg: str) -> bool:
    s = (msg or "").lower()
    return ("429" in s) or ("quota" in s) or ("rate limit" in s) or ("exceeded" in s)

def _is_not_found(msg: str) -> bool:
    return "404" in (msg or "") or "not found" in (msg or "").lower()

def _clean_candidates(cands: List[str]) -> List[str]:
    seen, ordered = set(), []
    for c in cands:
        if c in seen: continue
        if "exp" in c or "research" in c: continue
        if "flash" in c:
            seen.add(c); ordered.append(c)
    for c in cands:
        if c in seen: continue
        if "exp" in c or "research" in c: continue
        if "pro" in c:
            seen.add(c); ordered.append(c)
    for c in cands:
        if c in seen: continue
        seen.add(c); ordered.append(c)
    return ordered

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    # Live model discovery
    chat_models: List[str] = []
    embed_models: List[str] = []
    with st.expander("🧪 List available Gemini models (for this API key)", expanded=False):
        if not API_KEY:
            st.info("No GOOGLE_API_KEY/GEMINI_API_KEY found — you can still use local embeddings for RAG.\n"
                    "Add a key to use Gemini chat and Gemini embeddings.")
        try:
            if API_KEY:
                models = list(genai.list_models())
                all_rows = []
                for m in models:
                    name = getattr(m, "name", "")
                    caps = list(getattr(m, "supported_generation_methods", []) or [])
                    all_rows.append(f"- {name}  •  caps={','.join(caps) if caps else '(none)'}")
                    if "generateContent" in caps:
                        chat_models.append(name)
                    if "embedContent" in caps:
                        embed_models.append(name)
                all_rows.sort()
                st.code("\n".join(all_rows) if all_rows else "(none)")
        except Exception as e:
            st.error(f"ListModels failed: {e}")

    chat_models = _clean_candidates(sorted(set(chat_models))) or [
        "models/gemini-1.5-flash", "models/gemini-2.0-flash", "models/gemini-2.5-pro"
    ]
    embed_models = sorted(set(embed_models)) or ["models/text-embedding-004"]

    chat_model_id = st.selectbox("Chat model", options=chat_models, index=0)
    st.caption(f"Chat model: `{chat_model_id}`")

    # Embedding backend choice
    embed_backend = st.radio(
        "Embeddings backend",
        ["Local (MiniLM-L6-v2)", "Gemini API"],
        index=0,
        help="Use Local to avoid API quotas. Gemini API requires embeddings quota."
    )
    if embed_backend == "Gemini API":
        EMBED_MODEL = st.selectbox("Embedding model", options=embed_models, index=0)
        st.caption(f"Embedding model: `{EMBED_MODEL}`")
    else:
        EMBED_MODEL = "local/all-MiniLM-L6-v2"
        st.caption("Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (CPU)")

    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("TEMPERATURE", "0.3")), 0.05)
    max_tokens = st.slider("Max output tokens", 128, 4096, 1000, 64)

    style = st.radio("Answer style", ["Concise", "Detailed"], index=1, horizontal=True)
    target_words = st.slider("Target length (words)", 100, 1500, 600, 50)

    top_k = st.slider("Top-k chunks", 3, 15, 8, 1)
    chunk_size = st.slider("Chunk size (chars)", 500, 2400, 1200, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 600, 180, 10)

    sys_prompt = st.text_area(
        "System prompt",
        value=(
            "You are a helpful assistant. Prefer clarity and completeness.\n"
            "When context excerpts are provided, answer using ONLY that information. "
            "If not enough info is present, say: not sure."
        ),
        height=130,
    )

    rag_mode = st.checkbox("Use document context (RAG)", value=True,
                           help="If off or no excerpts found, assistant answers normally.")

    st.markdown("---")
    st.subheader("⏱️ Rate limit handling (Gemini)")
    auto_fallback = st.checkbox("Auto-fallback to other chat models on 404/429", value=True)
    max_retries = st.slider("Max retries (per model)", 0, 5, 2, 1)
    backoff_base = st.slider("Backoff base (seconds)", 1.0, 10.0, 2.0, 0.5)

    st.markdown("---")
    st.subheader("📄 Upload PDFs")
    uploaded_pdfs = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    st.caption("Indexing new files may take a moment (local embeddings are fastest).")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🧹 Clear chat"):
            st.session_state.pop("messages", None)
            st.toast("Chat cleared.")
    with colB:
        if st.button("♻️ Clear vector index"):
            for k in ("vs", "doc_map", "active_docs"):
                st.session_state.pop(k, None)
            st.toast("Index cleared.")
    with colC:
        st.write("")

    st.markdown("---")
    st.subheader("💾 Session")
    if "messages" in st.session_state and st.session_state.get("messages"):
        export_json = json.dumps({"messages": st.session_state["messages"]}, ensure_ascii=False, indent=2)
        st.download_button("⬇️ Download chat (JSON)", data=export_json.encode("utf-8"),
                           file_name="chat_session.json", mime="application/json")
    load_file = st.file_uploader("Load chat (JSON)", type=["json"], accept_multiple_files=False, key="load_chat")
    if load_file:
        try:
            loaded = json.loads(load_file.read().decode("utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("messages"), list):
                st.session_state["messages"] = loaded["messages"]
                st.toast("Chat loaded.")
            else:
                st.warning("Invalid chat JSON (expected {'messages': [...]}).")
        except Exception as e:
            st.error(f"Failed to load chat: {e}")

# =========================
# State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "doc_map" not in st.session_state:
    st.session_state.doc_map: Dict[str, Dict[str, Any]] = {}
if "active_docs" not in st.session_state:
    st.session_state.active_docs = set()

# =========================
# Embedding backends
# =========================
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible embeddings via google-generativeai, with retry/backoff."""
    def __init__(self, model: str = "models/text-embedding-004"):
        if not API_KEY:
            raise RuntimeError("Gemini embeddings require GOOGLE_API_KEY / GEMINI_API_KEY.")
        self.model = model

    def _embed_once(self, text: str) -> List[float]:
        resp = genai.embed_content(model=self.model, content=text or "")
        return resp["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            attempt = 0
            while True:
                try:
                    out.append(self._embed_once(t))
                    break
                except Exception as e:
                    msg = str(e)
                    if attempt >= max_retries or not _is_quota_or_rate(msg):
                        raise
                    delay = max(backoff_base * (2 ** attempt), _parse_retry_seconds(msg, backoff_base))
                    time.sleep(delay)
                    attempt += 1
        return out

    def embed_query(self, text: str) -> List[float]:
        attempt = 0
        while True:
            try:
                return self._embed_once(text)
            except Exception as e:
                msg = str(e)
                if attempt >= max_retries or not _is_quota_or_rate(msg):
                    raise
                delay = max(backoff_base * (2 ** attempt), _parse_retry_seconds(msg, backoff_base))
                time.sleep(delay)
                attempt += 1

class LocalEmbeddings(Embeddings):
    """LangChain-compatible local embeddings (no API, no quotas)."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)[0].tolist()

def get_embedder(choice: str):
    if choice == "Gemini API":
        return GeminiEmbeddings(model=EMBED_MODEL)
    return LocalEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

# =========================
# PDF → Documents → Index
# =========================
def pdf_to_documents(file_bytes: bytes, filename: str) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"source": filename, "page": i}))
    return docs

def add_pdf_to_index(file_bytes: bytes, filename: str):
    base_docs = pdf_to_documents(file_bytes, filename)
    if not base_docs:
        st.warning(f"No extractable text in {filename}")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(base_docs)

    doc_id = f"{filename}|{len(file_bytes)}"
    for d in chunks:
        m = d.metadata or {}
        d.metadata = {**m, "doc_id": doc_id, "source": filename, "page": int(m.get("page") or 0)}

    embedder = get_embedder(embed_backend)

    if st.session_state.vs is None:
        st.session_state.vs = FAISS.from_documents(chunks, embedder)
    else:
        st.session_state.vs.add_documents(chunks, embedder)

    st.session_state.doc_map[doc_id] = {"name": filename}
    st.session_state.active_docs.add(doc_id)

def maybe_index_new_uploads(files):
    if not files:
        return
    for f in files:
        doc_id = f"{f.name}|{f.size}"
        if doc_id not in st.session_state.doc_map:
            add_pdf_to_index(f.read(), f.name)

maybe_index_new_uploads(uploaded_pdfs)

# =========================
# Per-document toggles
# =========================
if st.session_state.doc_map:
    st.subheader("Included documents")
    all_ids = list(st.session_state.doc_map.keys())
    defaults = [i for i in all_ids if i in st.session_state.active_docs]
    selected = st.multiselect(
        "Choose which documents to include in retrieval:",
        options=all_ids,
        default=defaults,
        format_func=lambda i: st.session_state.doc_map[i]["name"],
    )
    st.session_state.active_docs = set(selected)

# =========================
# Retrieval
# =========================
def highlight_snippet(text: str, query: str, window: int = 520) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(w) >= 3]
    t = text or ""
    tl = t.lower()
    loc = -1
    if words:
        locs = [tl.find(w) for w in words if tl.find(w) != -1]
        loc = min(locs) if locs else -1
    snippet = t[:window] if loc == -1 else t[max(0, loc - window // 2): max(0, loc - window // 2) + window]
    for w in sorted(set(words), key=len, reverse=True):
        snippet = re.sub(fr"(?i)\b({re.escape(w)})\b", r"**\1**", snippet)
    return snippet.strip()

def retrieve(query: str, k: int) -> Tuple[str, List[Tuple[str, int, float]]]:
    if st.session_state.vs is None or not st.session_state.active_docs:
        return "", []
    raw = st.session_state.vs.similarity_search_with_score(query, k=k * 3)
    scored = []
    if raw:
        max_dist = max(d for _, d in raw) if raw else 1.0
        for doc, dist in raw:
            meta = doc.metadata or {}
            if meta.get("doc_id") in st.session_state.active_docs:
                sim = 1.0 - (dist / (max_dist + 1e-9))
                scored.append((doc, float(sim)))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    blocks, prov = [], []
    for rank, (doc, sim) in enumerate(top, start=1):
        meta = doc.metadata or {}
        name = meta.get("source", "(pdf)")
        page = int(meta.get("page") or 0)
        snippet = highlight_snippet(doc.page_content, query)
        prov.append((name, page, sim))
        blocks.append(f"[{rank}] {name} — p.{page} (score={sim:.3f})\n{snippet}")

    context = "=== RETRIEVED EXCERPTS ===\n" + "\n\n-----\n\n".join(blocks) if blocks else ""
    return context, prov

# =========================
# Gemini chat (direct) + streaming with retry/fallback
# =========================
def length_guidance_text() -> str:
    return (
        f"Write a {'brief' if style == 'Concise' else 'comprehensive'} answer. "
        f"Target ~{target_words} words. Use clear structure "
        f"({'bullet points' if style == 'Concise' else 'paragraphs + bullet points'}). "
        "Cite page numbers inline like (p.12)."
    )

def build_prompt(context: str, question: str, use_context: bool) -> str:
    if use_context and context.strip():
        return (
            "Use ONLY the excerpts below to answer. If the answer isn't clearly present, reply exactly: not sure.\n\n"
            f"{length_guidance_text()}\n\n"
            f"{context}\n\n"
            f"Question: {question}"
        )
    else:
        return (
            f"{length_guidance_text()}\n\n"
            "You are a helpful assistant. Answer the user's question directly. "
            "If the user greets you, greet them back naturally. "
            "If you don't have enough information, ask a brief clarifying question.\n\n"
            f"User: {question}"
        )

def make_model(model_name: str, system_prompt: str):
    if not API_KEY:
        raise RuntimeError("Gemini chat requires GOOGLE_API_KEY / GEMINI_API_KEY.")
    return genai.GenerativeModel(
        model_name=model_name,  # already "models/..." from ListModels or fallback list
        system_instruction=system_prompt,
        generation_config={"temperature": float(temperature), "max_output_tokens": int(max_tokens)},
    )

async def stream_with_retry_and_fallback(
    primary_model_id: str,
    fallback_pool: List[str],
    user_text: str,
    write_fn,
) -> str:
    tried, last_err = [], None
    pool = _clean_candidates([primary_model_id] + [m for m in fallback_pool if m != primary_model_id])

    for model_name in pool:
        tried.append(model_name)
        if not API_KEY:
            return "⚠️ Gemini chat requires an API key. You can still use local embeddings for RAG."
        model = make_model(model_name, sys_prompt)

        for attempt in range(max_retries + 1):
            acc = ""
            try:
                resp = model.generate_content(user_text, stream=True)
                for event in resp:
                    if hasattr(event, "text") and event.text:
                        acc += event.text
                        write_fn(acc)
                if hasattr(resp, "resolve"):
                    resp.resolve()
                return acc
            except Exception as e:
                err = str(e); last_err = err
                write_fn(f"{acc}\n\n_(retry {attempt+1}/{max_retries} on {model_name}: {err})_")
                if _is_not_found(err) and auto_fallback:
                    break
                if _is_quota_or_rate(err) and attempt < max_retries:
                    delay = max(backoff_base * (2 ** attempt), _parse_retry_seconds(err, backoff_base))
                    await asyncio.sleep(delay); continue
                if auto_fallback: break
                return f"⚠️ Error: {err}"

    return f"⚠️ Error after trying [{ ' → '.join(tried) }]: {last_err or 'unknown'}"

# =========================
# Chat UI
# =========================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask a question about your selected PDFs (or chat normally)…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    have_docs = st.session_state.vs is not None and bool(st.session_state.active_docs)
    context, prov = retrieve(user_input, k=top_k) if have_docs else ("", [])

    use_ctx = bool(rag_mode and context.strip())
    prompt_text = build_prompt(context, user_input, use_ctx)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        if API_KEY:
            try:
                fallback_pool = [m for m in (chat_models or []) if "exp" not in m.lower()]
                answer = asyncio.run(
                    stream_with_retry_and_fallback(
                        primary_model_id=chat_model_id,
                        fallback_pool=fallback_pool,
                        user_text=prompt_text,
                        write_fn=placeholder.markdown,
                    )
                )
            except RuntimeError:
                # Non-async fallback (rare in Streamlit)
                try:
                    model = make_model(chat_model_id, sys_prompt)
                    resp = model.generate_content(prompt_text)
                    answer = resp.text or ""
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                placeholder.markdown(answer)
        else:
            # No API key: do a simple local "response"
            answer = "I can index PDFs and retrieve snippets locally. To generate AI answers, add a GOOGLE_API_KEY."
            placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if use_ctx and isinstance(answer, str) and answer.strip():
        with st.expander("🔎 Context used (top excerpts with page numbers)"):
            st.markdown("The answer above was generated **only** from the following excerpts.")
            st.code(context[:8000])
