# Chat with your PDFs

A production-grade RAG application — upload PDFs, ask natural language questions, get answers with exact source citations and page numbers.

<!-- GIF demo goes here -->

## Tech stack

| Layer | Library | Why |
|---|---|---|
| PDF extraction | `pdfplumber` | Better table handling than PyPDF2 |
| Text splitting | `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| Embeddings | `openai` text-embedding-3-small | Cheapest, best quality for retrieval |
| Vector store | `chromadb` | Runs locally, no server needed |
| LLM chain | `langchain` + `openai` GPT-4o-mini | Low cost, fast, good enough |
| UI | `streamlit` | Ships fastest, looks clean |
| Deployment | Streamlit Cloud (free tier) | Live demo link in 5 minutes |

## Quickstart

```bash
git clone <repo>
cd chat-with-pdf
pip install -r requirements.txt

cp .env.example .env
# add your OPENAI_API_KEY to .env

streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload any PDF(s), click **Ingest / Reload**, then ask questions.

## Project structure

```
chat-with-pdf/
├── app.py                  ← Streamlit entry point
├── rag/
│   ├── ingestor.py         ← PDF load + chunk + embed + store
│   ├── retriever.py        ← MMR similarity search
│   └── chain.py            ← ConversationalRetrievalChain
├── utils/
│   └── helpers.py          ← chunking config, token counters
├── examples/
│   └── sample.pdf          ← demo file
├── requirements.txt
└── .env.example
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select the repo.
3. Add `OPENAI_API_KEY` in **Settings → Secrets**.
4. Click **Deploy** — live in ~60 seconds.

## How it works

1. **Ingest** — `pdfplumber` extracts text (+ tables) page-by-page; `RecursiveCharacterTextSplitter` cuts into 800-token chunks with 100-token overlap; OpenAI embeds each chunk; ChromaDB stores vectors locally.
2. **Retrieve** — MMR search fetches 20 candidates, reranks for diversity, returns top 5.
3. **Generate** — `ConversationalRetrievalChain` condenses the chat history into a standalone question, retrieves context, then streams a grounded answer via GPT-4o-mini.
4. **Cite** — `return_source_documents=True` gives back the chunks; the UI surfaces filename + page number under each answer.
