# 📄 AskMyDocs — Chat with your documents using RAG

AskMyDocs lets you upload PDFs or text files and ask questions in plain English. Every answer is **grounded in your documents and cited down to the page**, so you can verify it instead of trusting a black box.

It tackles a common problem: people waste hours searching long handbooks, contracts, research papers and manuals for one fact. A plain LLM will happily make up an answer; AskMyDocs retrieves the relevant passages first and only answers from them.

> **Live demo:** _add your Streamlit Cloud link here_
> **Demo GIF:** _add `docs/demo.gif` here_

---

## ✨ Features

- **Multi-document Q&A** over PDF, TXT and Markdown files
- **Page-level citations** — every claim links to `[n]` with file name and page number
- **Hybrid retrieval** — dense vector search (FAISS) + keyword search (BM25), fused with Reciprocal Rank Fusion
- **Hallucination guardrail** — the model says *"I couldn't find that"* instead of guessing
- **Follow-up questions** — recent chat history is passed in for conversational context
- **Streaming answers** and an expandable **Sources** panel showing why each excerpt matched (semantic vs keyword)
- **Google Gemini** — Gemini embeddings and chat generation with a free-tier API key
- **Offline test suite** — fake embedder/LLM so tests run without an API key

## 🏗️ How it works

```
Upload ──► Parse (pypdf) ──► Sentence-aware chunking ──► Embeddings ──► FAISS index
                                     (page numbers kept)           └──► BM25 index

Question ──► Embed query ──► FAISS top-N ─┐
         └─► Tokenize ─────► BM25 top-N ──┴─► Reciprocal Rank Fusion ──► top-k excerpts
                                                                              │
                          LLM (Gemini) ◄── grounded prompt + citations┘
                                     │
                          Streamed answer with [1][2] citations + Sources panel
```

**Why hybrid search?** Embeddings are great at meaning ("time off" ≈ "annual leave") but weak on exact tokens like policy numbers, acronyms and names. BM25 is the opposite. RRF combines both rankings without needing to calibrate their different score scales.

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| LLM & embeddings | Google Gemini (`gemini-3.8-flash`, `gemini-embedding-001`) |
| Vector search | FAISS (`IndexFlatIP`, cosine similarity) |
| Keyword search | BM25 (`rank-bm25`) |
| Document parsing | pypdf |
| UI | Streamlit |
| Testing | pytest |

## 🚀 Run it locally (VS Code)

**Prerequisites:** Python 3.10–3.12 and [VS Code](https://code.visualstudio.com/) with the Python extension.

1. **Open the folder** in VS Code (`File → Open Folder… → askmydocs`).

2. **Create a virtual environment** in the VS Code terminal (`` Ctrl+` ``):

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   If VS Code asks *"Select interpreter"*, choose the one inside `.venv`.

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key.** Copy `.env.example` to `.env` and paste your key:

   ```bash
   # Windows
   copy .env.example .env
   # macOS / Linux
   cp .env.example .env
   ```

   Get a free Gemini key at <https://aistudio.google.com/apikey> and add it as `GEMINI_API_KEY` in `.env`.

5. **Run the app** — either press **F5** (uses the included *Run AskMyDocs* launch config) or:

   ```bash
   streamlit run app.py
   ```

   It opens at <http://localhost:8501>. Upload `sample_docs/sample_employee_handbook.pdf` and try:
   - *How many days of annual leave do I get, and can I carry them forward?*
   - *What's the notice period for interns?*
   - *Do I need a VPN when working from home?*
   - *What is the company's stock option policy?* → should answer that it couldn't find it

6. **Run the tests** (no API key needed):

   ```bash
   pytest -q
   ```

## ☁️ Deploy (free) on Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to <https://share.streamlit.io> → **New app** → pick your repo, branch `main`, file `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy, then paste the URL at the top of this README.

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "AskMyDocs: RAG document Q&A with hybrid retrieval and citations"
git branch -M main
git remote add origin https://github.com/<your-username>/askmydocs.git
git push -u origin main
```

## 📁 Project structure

```
askmydocs/
├── app.py                 # Streamlit UI
├── rag/
│   ├── config.py          # settings from .env
│   ├── loader.py          # PDF/TXT/MD → pages
│   ├── chunker.py         # sentence-aware chunks with overlap, keeps page numbers
│   ├── providers.py       # Gemini embedding + chat wrappers
│   ├── retriever.py       # FAISS + BM25 hybrid search with RRF
│   └── pipeline.py        # orchestration + grounded prompt
├── tests/test_rag.py      # offline tests with fake models
├── sample_docs/           # demo PDF
├── .vscode/               # run/debug config, interpreter + pytest settings
├── requirements.txt
└── .env.example
```

## 🔭 Possible next steps

- Cross-encoder re-ranking of the fused results
- Persisting indexes to disk so large document sets aren't re-embedded
- An evaluation set (question → expected page) to measure retrieval recall@k
- OCR for scanned PDFs
