# RAG Document Q&A API

A production-ready Retrieval-Augmented Generation (RAG) system with advanced document ingestion, hybrid search, reranking, and AI-powered Q&A. Built with FastAPI, Qdrant, and OpenAI.

## Features

### Core RAG Capabilities
- **Hybrid Search**: Combines dense embeddings (Sentence-Transformers) with sparse embeddings (SPLADE) for superior retrieval
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to rerank retrieved documents for better relevance
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve retrieval quality
- **Streaming Responses**: Real-time token streaming via Server-Sent Events (SSE)
- **Multi-turn Conversations**: Maintains conversation context with history support

### Data Source Connectors
| Source | Description | Status |
|--------|-------------|--------|
| **File Upload** | PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, HTML, RTF, Images (OCR) | Implemented |
| **Web Crawler** | URLs with Playwright browser support for JS-heavy sites | Implemented |
| **Git Repository** | Clone and index repos | Implemented |
| **Confluence** | Atlassian Cloud integration | Implemented |
| **SharePoint** | Microsoft Graph API | Implemented |
| **Notion** | Database and page ingestion | Implemented |
| **Database** | PostgreSQL, MongoDB | Implemented |
| **API** | Generic REST API connector | Implemented |

### Advanced Features
| Feature | Description | Implementation |
|---------|-------------|----------------|
| **Reranking** | Cross-encoder reranking | `ms-marco-MiniLM-L-6-v2` |
| **Hybrid Search** | Dense + Sparse vectors | SPLADE + Sentence-Transformers |
| **HyDE** | Query expansion | GPT-generated hypothetical docs |
| **Metadata Filtering** | Filter by document attributes | Qdrant filters |
| **Multiple Embeddings** | Provider flexibility | HuggingFace, OpenAI |
| **Evaluation Metrics** | P@K, R@K, NDCG, MRR | Built-in evaluation suite |
| **Async Job Processing** | Background ingestion | MongoDB job persistence |
| **Browser Scraping** | JavaScript-heavy sites | Playwright + Chromium |
| **OCR Support** | Scanned PDFs & images | Tesseract + pdf2image |
| **Markitdown** | Enhanced document extraction | Microsoft markitdown |

## Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **Vector Database**: Qdrant
- **Dense Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Sparse Embeddings**: FastEmbed SPLADE (`prithivida/Splade_PP_en_v1`)
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Job Persistence**: MongoDB (optional)

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- OpenAI API Key

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag-qa-api
```

### 2. Configure Environment

Create a `.env` file:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional - Embedding Configuration
EMBEDDING_PROVIDER=huggingface  # or "openai"
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Optional - Job Persistence
MONGODB_URL=mongodb://localhost:27017/rag_chat
```

### 3. Run with Docker Compose

```bash
docker-compose up --build
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Qdrant UI**: http://localhost:6333/dashboard

### 4. Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker-compose up qdrant -d

# Run API
uvicorn app.main:app --reload --port 8000
```

## API Reference

### Chat & Q&A

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/query` | POST | Single question with RAG retrieval |
| `/chat/query/stream` | POST | Streaming response with SSE |
| `/chat/` | POST | Multi-turn conversation with history |
| `/chat/stream` | POST | Streaming conversation |
| `/chat/models` | GET | List available LLM models |
| `/chat/health` | GET | RAG pipeline health check |

**Query Request Example:**
```json
{
  "question": "What is machine learning?",
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 1000,
  "use_hyde": false,
  "metadata_filters": {"source": "docs"},
  "score_threshold": 0.5
}
```

**Streaming Response Events:**
```
data: {"type": "retrieval_start", "data": {"question": "..."}}
data: {"type": "retrieval_complete", "data": {"num_docs": 5, "sources": [...]}}
data: {"type": "generation_start", "data": {}}
data: {"type": "token", "data": {"content": "Machine"}}
data: {"type": "token", "data": {"content": " learning"}}
data: {"type": "done", "data": {}}
```

### Document Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest/upload` | POST | Upload and ingest a file |
| `/ingest/file` | POST | Ingest from local path |
| `/ingest/run` | POST | Synchronous ETL ingestion |
| `/ingest/submit` | POST | Async job submission |
| `/ingest/status/{job_id}` | GET | Get job status |
| `/ingest/jobs` | GET | List recent jobs |
| `/ingest/jobs/{job_id}/logs` | GET | Get job logs |

**ETL Request Example:**
```json
{
  "source_type": "confluence",
  "source_params": {
    "base_url": "https://company.atlassian.net/wiki",
    "email": "user@company.com",
    "api_token": "your-api-token"
  },
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "store_in_qdrant": true
}
```

**Supported Source Types:**
- `file` - Local file or directory (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, RTF, images)
- `web` - Web URL with crawling (supports Playwright browser mode for JS-heavy sites)
- `git` - Git repository
- `confluence` - Atlassian Confluence
- `sharepoint` - Microsoft SharePoint
- `notion` - Notion workspace
- `database` - PostgreSQL, MongoDB
- `api` - REST API endpoints

**Web Source with Browser Mode (for JavaScript-heavy sites like Yahoo Finance):**
```json
{
  "source_type": "web",
  "source_params": {
    "url": "https://finance.yahoo.com/quote/AAPL",
    "use_browser": true,
    "wait_time": 5000,
    "auto_dismiss_popups": true
  }
}
```

**File Source with OCR and Password Support:**
```json
{
  "source_type": "file",
  "source_params": {
    "file_path": "/path/to/document.pdf",
    "ocr_enabled": true,
    "ocr_language": "eng",
    "password": "optional-pdf-password"
  }
}
```

### Integrations Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/integrations/` | POST | Create saved integration |
| `/integrations/` | GET | List all integrations |
| `/integrations/{id}` | DELETE | Delete integration |

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Semantic search without LLM |

### Collection Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collection/info` | GET | Get collection details |
| `/collection/create` | POST | Create/recreate collection |
| `/collection` | DELETE | Delete collection |

### Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluation/benchmark` | POST | Run evaluation benchmark |
| `/evaluation/metrics` | GET | Get aggregated metrics |
| `/evaluation/feedback` | POST | Submit user feedback |
| `/evaluation/export/{type}` | GET | Export metrics to CSV |

**Available Metrics:**
- Retrieval: Precision@K, Recall@K, NDCG@K, MRR
- QA: Exact Match, Contains Match, LLM-as-Judge

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/` | GET | API info |

## Project Structure

```
rag-qa-api/
├── app/
│   ├── main.py                 # Application entry point
│   ├── models/
│   │   ├── vector_store.py     # Qdrant integration with hybrid search
│   │   ├── job.py              # Job persistence model
│   │   └── integration.py      # Integration model
│   ├── routers/
│   │   ├── chat.py             # Chat & Q&A endpoints
│   │   ├── ingestion.py        # Document ingestion
│   │   ├── search.py           # Search endpoints
│   │   ├── collection.py       # Collection management
│   │   ├── evaluation.py       # Evaluation & metrics
│   │   └── integrations.py     # Integration management
│   └── services/
│       ├── rag_pipeline.py     # Core RAG logic with reranking
│       ├── etl_pipeline.py     # Document processing
│       ├── evaluation.py       # Metric calculations
│       ├── metrics_logger.py   # Metrics persistence
│       ├── embedding_providers.py  # HuggingFace/OpenAI embeddings
│       └── data_sources/
│           ├── base.py         # Abstract base class
│           ├── file_source.py
│           ├── web_source.py
│           ├── git_source.py
│           ├── confluence_source.py
│           ├── sharepoint_source.py
│           ├── notion_source.py
│           ├── database_source.py
│           └── api_source.py
├── tests/
│   └── test_*.py               # Unit and integration tests
├── logs/
│   └── metrics/                # Query and feedback logs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `MONGODB_URL` | None | MongoDB for job persistence |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | 5 | Number of documents to retrieve |
| `temperature` | float | 0.7 | LLM sampling temperature |
| `max_tokens` | int | 1000 | Maximum response tokens |
| `use_hyde` | bool | false | Enable HyDE query expansion |
| `score_threshold` | float | None | Minimum relevance score |
| `metadata_filters` | dict | None | Filter by metadata fields |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chat_endpoints.py -v
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│  rag-chat-ui    │────▶│   rag-qa-api    │
│   (React)       │     │   (Backend)     │     │   (This API)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                ▼
                ┌───────────────┐              ┌─────────────────┐              ┌─────────────────┐
                │    Qdrant     │              │    OpenAI       │              │    MongoDB      │
                │ Vector Store  │              │    GPT-4o       │              │  Job Storage    │
                └───────────────┘              └─────────────────┘              └─────────────────┘
```

## RAG Pipeline Flow

```
1. Query Received
       │
       ▼
2. [Optional] HyDE: Generate hypothetical answer
       │
       ▼
3. Embed Query (Dense + Sparse)
       │
       ▼
4. Hybrid Search in Qdrant
   - Dense: Sentence-Transformer embeddings
   - Sparse: SPLADE embeddings
       │
       ▼
5. Cross-Encoder Reranking
   - Score each (query, doc) pair
   - Sort by relevance
       │
       ▼
6. Build Prompt with Context
       │
       ▼
7. Generate Answer (OpenAI)
   - Streaming tokens via SSE
       │
       ▼
8. Return Response with Sources
```

## License

MIT License
