# 📄 Ask Your PDF — Local AI Document Assistant

Ask questions about your PDF documents using a fully local AI pipeline.  
No OpenAI API, no cloud upload — everything runs on your machine.

---

## 🚀 Overview

**Ask Your PDF** is a Retrieval-Augmented Generation (RAG) application that allows users to:

- Upload a PDF
- Ask natural language questions
- Get answers strictly based on document content

The system combines:
- Local embeddings
- Vector search (FAISS)
- Local LLM (Ollama)

---

## 🧠 How It Works

1. 📂 Upload a PDF document  
2. 📄 Text is extracted page-by-page  
3. ✂️ Text is split into smaller sections  
4. 🔍 Sections are embedded using a local model  
5. 📦 Stored in FAISS vector database  
6. ❓ User asks a question  
7. 🎯 Relevant sections are retrieved  
8. 🤖 Local LLM generates answer using only document context  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **LangChain** – RAG pipeline
- **FAISS** – vector database
- **HuggingFace Embeddings** – local embeddings
- **Ollama (phi3:mini)** – local LLM
- **pdfplumber** – PDF parsing

---

## 🔒 Privacy First

- ✅ Runs fully locally  
- ✅ No API keys required  
- ✅ No data sent to external servers  
- ✅ Your documents stay on your machine  

---

## ▶️ Getting Started

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/ask-your-pdf.git
cd ask-your-pdf

2. Install dependencies

pip install -r requirements.txt

3. Install Ollama

Download and install:

👉 https://ollama.com

4. Pull the model

ollama pull phi3:mini

5. Run the app

streamlit run ask_your_pdf.py

⚙️ Configuration

Increase PDF upload limit:

Create:

.streamlit/config.toml

[server]
maxUploadSize = 1024

## 🎯 Features

- 💬 Chat-style interface  
- 📄 PDF upload and parsing  
- 🔍 Adjustable document search depth  
- 📊 Page-aware answers  
- 🧠 Local LLM inference  
- ⚡ Cached embeddings and models  
- 🧪 Optional context debugging  

---

## ⚠️ Limitations

- Small models may occasionally hallucinate  
- Large PDFs may take time to process initially  
- Performance depends on system RAM and CPU  

---

## 📈 Future Improvements

- Multi-document support  
- Persistent vector database (Chroma)  
- Better reranking for accuracy  
- Chat memory across sessions  
- Deployment as web service  

---

## 📚 What I Learned

- RAG architecture (retrieval + generation)  
- Vector databases (FAISS)  
- Embeddings and semantic search  
- Prompt engineering for hallucination control  
- Local LLM limitations and tuning  
- Streamlit UI design  
- Building production-style ML applications  

---

## 🧑‍💻 Author

Built as a portfolio project demonstrating practical LLM + RAG system design.
