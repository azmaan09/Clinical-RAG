# Clinical RAG API

A production-ready FastAPI service for querying patient medical records using Retrieval-Augmented Generation (RAG).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│  POST /api/query         →  RAG Pipeline                    │
│  POST /api/ingest/pdf    →  PDF Ingestion                   │
│  POST /api/ingest/text   →  Text Ingestion                  │
│  GET  /api/health        →  Health Check                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                         ┌─────────────────┐
│  Milvus       │                         │  Gemini API     │
│  (Zilliz)     │                         │                 │
│               │                         │  - Embeddings   │
│  Vector Store │                         │  - Generation   │
└───────────────┘                         └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd rag-api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

**Option A: Using GCP Secret Manager** (recommended for production)
```bash
# Set your GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

**Option B: Using Environment Variables** (for local development)
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run the Server

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the patient diagnosis?", "top_k": 3}'
```

**Response:**
```json
{
  "question": "What is the patient diagnosis?",
  #+ Clinical RAG — Human-centered Retrieval-Augmented Generation for Clinical Records

  Clinical RAG is a small, pragmatic project that demonstrates a production-oriented
  Retrieval-Augmented Generation (RAG) workflow for querying clinical / patient records.
  It combines a FastAPI backend (the RAG API) with a Streamlit front-end for quick
  exploration, and uses a vector store (Milvus) plus a modern LLM for embeddings and
  generation (the project refers to Gemini in examples).

  Why this project exists
  - Provide a reproducible template for ingesting clinical documents (PDFs / text),
    indexing them as vectors, and answering natural-language questions with cited
    sources (document chunks).
  - Show how to wire ingestion, vector search, and generation together with
    clear APIs and a minimal UI.

  Important: This repository contains example wiring and convenience scripts. If you
  intend to use this with real patient data, ensure you follow your organization’s
  privacy, security, and compliance policies (HIPAA, GDPR, etc.).

  Highlights
  - FastAPI endpoints for ingestion, querying, and health checks
  - Streamlit UI for quick manual queries and PDF uploads (`streamlit_app.py`)
  - Milvus vector store for scalable similarity search
  - Designed for easy replacement of the embedding / generation model

  Table of contents
  - Features
  - Quick start (local)
  - Environment & configuration
  - Running the app (API + Streamlit)
  - Project layout
  - Deployment notes
  - Contributing and support

  Features
  - Ingest PDFs or plain text into the vector store with chunking and overlap
  - Query the collection with a single REST call, returning an answer and
    the source chunks used to generate it (score + snippet)
  - Streamlit frontend that calls the backend and shows results with source snippets

  Quick start — run locally (recommended for evaluation)

  1) Create a virtual environment and install dependencies

  ```bash
  cd "/Users/amankhan/Downloads/final project/clinical-rag"
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  2) Configure environment

  Copy the example environment file and edit values as needed.

  ```bash
  cp .env.example .env
  # Edit .env to add credentials and configuration values
  ```

  If you use cloud secrets (e.g., GCP Secret Manager), set the appropriate
  environment variable(s) instead (see `.env.example`).

  3) Start the FastAPI backend

  ```bash
  uvicorn app.main:app --reload --port 8000
  ```

  4) Open the Streamlit UI (optional, for manual exploration)

  ```bash
  streamlit run streamlit_app.py
  ```

  By default the Streamlit app expects the API at http://localhost:8000/api. If
  you host the API elsewhere, update `API_BASE_URL` at the top of
  `streamlit_app.py` or provide a runtime config.

  API quick examples

  Query the RAG API (JSON request)

  ```bash
  curl -X POST "http://localhost:8000/api/query" \
    -H "Content-Type: application/json" \
    -d '{"question":"What is the patient diagnosis?","top_k":3}'
  ```

  Ingest a PDF (multipart form upload)

  ```bash
  curl -X POST "http://localhost:8000/api/ingest/pdf" -F "file=@patient_record.pdf"
  ```

  Project layout (high level)

  ```
  .
  ├── app/                 # FastAPI application code (API endpoints, services)
  ├── streamlit_app.py     # Lightweight Streamlit UI for manual testing
  ├── requirements.txt     # Python dependencies
  ├── .env.example         # Example environment variables
  └── README.md
  ```

  Configuration notes
  - Use `.env` (copied from `.env.example`) for local development.
  - For production, prefer a managed secret store (GCP Secret Manager is shown
    in examples) and ensure credentials are never checked into the repo.
  - The Milvus collection name, model IDs, and chunking parameters are configurable
    in the project settings (see `app/config.py`).

  Deployment
  - A minimal Dockerfile is present in the original README and works well for
    containerized deployment. For production, you should:
    - Run Milvus (or another vector DB) in a managed or containerized setup
    - Use a hosted LLM or managed inference endpoint with appropriate auth
    - Add an authentication layer and CORS restrictions to the API

  Security & compliance
  - This project is a demo / reference. Do not use unencrypted or unauthorized
    storage for protected health information (PHI).
  - Add transport-layer security (TLS), authentication, auditing, and access
    controls before using in production.

  Contributing
  - Bug reports and pull requests are welcome. If you open a PR, please include
    a brief description of the change and an update to this README if relevant.

  Maintainer
  - If you need help adapting this project to your environment, let the
    maintainer know the target infrastructure (Milvus version, LLM provider,
    and deployment platform) and they can provide guidance.

  License
  - This project is provided as-is for educational and prototyping purposes.
