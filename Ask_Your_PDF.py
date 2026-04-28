import hashlib

import pdfplumber
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Ask Your PDF"
APP_SUBTITLE = "Private local document assistant powered by RAG, FAISS, and Ollama."
OLLAMA_MODEL = "phi3:mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
    color: #1f2937;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Hide Streamlit file upload size helper text */
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: none !important;
}

.hero-card {
    background: #ffffff;
    padding: 28px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 35px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 6px;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    text-align: center;
}

.info-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 22px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

.stChatMessage {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


st.markdown(f"""
<div class="hero-card">
    <div class="main-title">📄 {APP_TITLE}</div>
    <div class="subtitle">{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("📂 Document")
    st.caption("Upload a PDF. The document is processed locally on your machine.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        st.success(f"✅ Loaded: {uploaded_file.name}")

    st.divider()

    st.success(f"🤖 Model: {OLLAMA_MODEL}")

    top_k = st.slider(
        "🔍 How much of the document to search",
        min_value=2,
        max_value=8,
        value=4,
        help="Higher values search more document sections. This may improve broad answers but can add noise.",
    )

    show_context = st.checkbox(
        "Show retrieved document context",
        value=False,
        help="Useful for debugging and verifying where the answer came from.",
    )

    st.divider()
    st.caption("No OpenAI API key required. Runs locally after models are installed.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_doc_hash" not in st.session_state:
    st.session_state.active_doc_hash = None


@st.cache_resource
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load local embedding model once and cache it."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_llm() -> Ollama:
    """Load local Ollama model wrapper once and cache it."""
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=0,
    )


def calculate_file_hash(file) -> str:
    """Create a stable hash for uploaded file content."""
    file.seek(0)
    file_bytes = file.read()
    file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()


def extract_pdf_pages(file) -> list[Document]:
    """Extract PDF text page-by-page and store page numbers as metadata."""
    documents = []

    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)
        progress_bar = st.progress(0)
        status = st.empty()

        for page_number, page in enumerate(pdf.pages, start=1):
            status.write(f"Reading page {page_number} of {total_pages}...")

            page_text = page.extract_text() or ""

            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": page_number},
                    )
                )

            progress_bar.progress(page_number / total_pages)

        status.empty()
        progress_bar.empty()

    return documents


@st.cache_resource(show_spinner=False)
def build_vector_index(_file, file_hash: str):
    """
    Build FAISS index from PDF pages.

    The uploaded file argument is prefixed with `_` so Streamlit does not try
    to hash it directly. `file_hash` controls cache invalidation.
    """
    raw_documents = extract_pdf_pages(_file)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1200,
        chunk_overlap=250,
    )

    chunks = text_splitter.split_documents(raw_documents)

    embeddings = load_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vector_store, chunks


def format_docs(docs: list[Document]) -> str:
    """Format retrieved chunks with page references."""
    formatted_docs = []

    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        formatted_docs.append(f"[Page {page}]\n{doc.page_content}")

    return "\n\n".join(formatted_docs)


def clean_output(text: str) -> str:
    """Remove common prompt echoes from local LLM output."""
    markers = [
        "Final answer:",
        "Answer:",
        "DOCUMENT CONTEXT:",
        "QUESTION:",
    ]

    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1]

    return text.strip()


prompt = ChatPromptTemplate.from_template(
    "You are a PDF extraction assistant, not a general chatbot.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the text inside DOCUMENT CONTEXT.\n"
    "- Do NOT use outside knowledge.\n"
    "- Do NOT invent facts.\n"
    "- Extract specific facts, numbers, dates, percentages, and comparisons from the document.\n"
    "- Include page numbers when possible.\n"
    "- Format the answer clearly using bullet points when useful.\n"
    "- If the answer is not clearly present, say exactly:\n"
    "  This information is not available in the uploaded document.\n\n"
    "DOCUMENT CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "Final answer:"
)


if uploaded_file is None:
    st.markdown("""
    <div class="info-card">
        <h3>👈 Upload a PDF to begin</h3>
        <p>
            Ask Your PDF will extract text, build a local vector index,
            and answer questions using only the uploaded document.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    current_doc_hash = calculate_file_hash(uploaded_file)

    if st.session_state.active_doc_hash != current_doc_hash:
        st.session_state.messages = []
        st.session_state.active_doc_hash = current_doc_hash

    with st.spinner("Building local document index..."):
        vector_store, chunks = build_vector_index(uploaded_file, current_doc_hash)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = load_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.markdown(f"""
    <div class="info-card">
        <h3>✅ Document ready</h3>
        <p><b>File:</b> {uploaded_file.name}</p>
        <p><b>Search sections created:</b> {len(chunks)}</p>
        <p><b>Local model:</b> {OLLAMA_MODEL}</p>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask something about the PDF...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        if show_context:
            retrieved_docs = retriever.invoke(user_question)
            with st.expander("Retrieved document context"):
                st.write(format_docs(retrieved_docs))

        with st.chat_message("assistant"):
            with st.spinner("Thinking locally..."):
                response = chain.invoke(user_question)
                final_answer = clean_output(response)
                st.markdown(final_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )