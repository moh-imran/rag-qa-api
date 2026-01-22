# RAG Document Q&A API

A complete Retrieval-Augmented Generation (RAG) system with document ingestion, semantic search, and AI-powered Q&A. Built with FastAPI, Qdrant, and OpenAI.

## 🚀 Features

- **Automated ETL Pipeline**: Extract text from PDF, DOCX, TXT, and Markdown files.
- **Semantic Vector Search**: Uses Qdrant for fast and accurate document retrieval.
- **AI-Powered Q&A**: Uses OpenAI's `gpt-4o-mini` to generate context-aware answers.
- **Local Embeddings**: High-quality local embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Flexible Chat Interface**: Supports single queries and full conversation history.
- **Docker Ready**: Easy deployment with Docker and Docker Compose.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o-mini
- **Vector Database**: Qdrant
- **Embeddings**: Sentence-Transformers
- **Environment**: Python 3.10+

## 📋 Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (optional, for running Qdrant)
- OpenAI API Key

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd onboarding
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Start Docker services** (for Qdrant):
   ```bash
   docker-compose up -d
   ```

## 🏃 Running the Application

Start the FastAPI server:
```bash
python -m app.main
```
The API will be available at `http://localhost:8000`.
Explore the interactive API documentation at `http://localhost:8000/docs`.

## 🔌 API Endpoints

### Ingestion
- `POST /ingest/file`: Ingest documents from a local file path or directory.
- `POST /ingest/upload`: Upload and ingest a single file.
 - `POST /ingest/run`: Generic synchronous ingestion entrypoint. Body: `source_type`, `source_params`, `chunk_size`, `chunk_overlap`, `batch_size`, `store_in_qdrant`.
 - `POST /ingest/submit`: Submit an asynchronous ingest job (returns `job_id`). Use `/ingest/status/{job_id}` to poll and `/ingest/jobs` to list recent jobs.

### Chat & Q&A
- `POST /chat/query`: Ask a question based on ingested documents.
- `POST /chat/`: Chat with conversation history.
- `GET /chat/models`: List available OpenAI models.

### Collection Management
- `GET /collection/info`: Get details about the vector collection.
- `POST /collection/create`: Create or recreate the collection.
- `DELETE /collection`: Delete the entire collection.

### System
- `GET /health`: Check system health and service connectivity.

## 📂 Project Structure

```text
├── app/
│   ├── models/          # Data models and Vector Store abstraction
│   ├── routers/         # API endpoints (Chat, Ingestion, etc.)
│   ├── services/        # Core logic (ETL and RAG pipelines)
│   └── main.py          # Application entry point
├── tests/               # Unit and integration tests
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
└── requirements.txt     # Python dependencies
```

## 🧪 Testing

Run tests using pytest:
```bash
pytest
```

## 🔌 New Data Source Connectors

This project includes minimal connectors for additional enterprise sources:

- Confluence (Atlassian Cloud): `app/services/data_sources/confluence_source.py` — submit a job with `source_type: 'confluence'` and `source_params: { base_url, email, api_token, space_key? or content_id? }`.
- SharePoint (Microsoft Graph): `app/services/data_sources/sharepoint_source.py` — submit a job with `source_type: 'sharepoint'` and `source_params: { site_id, access_token }`.

Notes:
- Confluence connector expects a Cloud site URL and an API token for a user. It fetches storage-format HTML and strips tags.
- SharePoint connector uses a Microsoft Graph access token. It currently downloads simple text-like files (txt, md, html). Extend for `.docx`/`.pdf` if needed.

## 🧾 Job Persistence

If you want ingest jobs to persist across restarts, set `MONGODB_URL` in your environment and install `motor` and `beanie` (they are included in `requirements.txt`). Example:

```env
MONGODB_URL=mongodb://localhost:27017/rag_qa
```

When `MONGODB_URL` is set, the API will persist job metadata/results to the `ingest_jobs` collection and expose `/ingest/jobs` and `/ingest/status/{job_id}` endpoints.
