# ResearchMind AI - Document Intelligence Pipeline

**Production-grade RAG backend for document analysis and intelligent Q&A**

## Project Overview

This is NOT a chatbot. This is a document intelligence pipeline similar to NotebookLM, built with professional backend architecture.

**Core Architecture:**
```
Documents → Parsing → Chunking → Embeddings → Vector Storage → Retrieval → LLM → Grounded Answer
```

## Quick Start

### Prerequisites
- Python 3.9+
- Qdrant (local or cloud)
- Google Gemini API key

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your keys
```

### Run Server

```bash
# Development
python -m app.main

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server runs on `http://localhost:8000`

## API Endpoints

### 1. Upload PDF Document
```bash
POST /api/v1/upload

# Using curl
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/v1/upload

# Response
{
  "filename": "document.pdf",
  "pages": 10,
  "chunks": 45,
  "status": "indexed",
  "document_id": "uuid",
  "file_size": 2048576,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Ask Questions
```bash
POST /api/v1/ask

{
  "question": "What are the main findings?",
  "document_id": "optional-uuid",
  "top_k": 5
}

# Response
{
  "question": "What are the main findings?",
  "answer": "According to the document...",
  "citations": [
    {
      "source": "document.pdf",
      "page": 4,
      "similarity_score": 0.89
    }
  ],
  "confidence": 0.87,
  "retrieval_stats": {
    "total_chunks": 5,
    "avg_score": 0.85
  }
}
```

### 3. List Documents
```bash
GET /api/v1/documents

# Response
{
  "total": 3,
  "documents": [...]
}
```

### 4. Health Check
```bash
GET /api/v1/health

# Response
{
  "status": "healthy",
  "vectordb": {
    "connected": true,
    "points_count": 150
  }
}
```

## Architecture

### Project Structure
```
backend/
├── app/
│   ├── api/              # FastAPI routes
│   │   ├── documents.py  # Upload/manage documents
│   │   └── questions.py  # Q&A endpoints
│   ├── rag/              # RAG pipeline
│   │   ├── ingestion.py  # PDF extraction (Stage 1)
│   │   ├── chunking.py   # Text chunking (Stage 2)
│   │   ├── embeddings.py # Embeddings (Stage 3)
│   │   ├── retrieval.py  # Retrieval (Stage 5)
│   │   └── answer_generation.py # Answer generation (Stage 6)
│   ├── vectorstore/      # Vector DB
│   │   └── qdrant_store.py # Qdrant integration (Stage 4)
│   ├── services/         # Business logic
│   │   ├── document_service.py
│   │   └── answer_service.py
│   ├── models/           # Pydantic schemas
│   ├── utils/            # Utilities
│   ├── config.py         # Settings
│   └── main.py           # App entry point
├── tests/                # Test suite
├── uploads/              # Uploaded PDFs (local)
├── requirements.txt      # Dependencies
├── .env                  # Configuration
└── README.md
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **Server** | Uvicorn |
| **Embeddings** | sentence-transformers |
| **Vector DB** | Qdrant |
| **LLM** | Google Gemini |
| **RAG Framework** | LangChain |
| **PDF Processing** | PyPDF2 |

## Key Features

### ✅ Implemented
- PDF ingestion with text extraction
- Semantic chunking with overlap
- Dense embeddings (sentence-transformers)
- Vector similarity search
- LLM-based answer generation
- Source citations with page numbers
- Confidence scoring
- Document management
- Health check endpoint

### 🚀 Roadmap
- [ ] LangGraph multi-agent workflows
- [ ] Streaming responses
- [ ] Chat memory (short-term)
- [ ] Document comparison
- [ ] Topic extraction
- [ ] Quiz generation
- [ ] PostgreSQL persistence
- [ ] Redis caching
- [ ] Advanced chunking strategies
- [ ] Hybrid search

## Configuration

### Environment Variables

```env
# Gemini API
GEMINI_API_KEY=your-api-key

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional for local
QDRANT_COLLECTION_NAME=research_docs

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Pipeline Explanation

### Stage 1: PDF Ingestion
- Extract text from PDFs
- Preserve page metadata
- Clean and normalize text

### Stage 2: Semantic Chunking
- Split text into ~1000 char chunks
- Maintain 200 char overlap for context
- Preserve page markers

### Stage 3: Embeddings
- Convert chunks to dense vectors
- Use sentence-transformers (384 dim)
- Batch processing for efficiency

### Stage 4: Vector Storage (Qdrant)
- Store embeddings + metadata
- Cosine similarity indexing
- Enable semantic search

### Stage 5: Retrieval
- Embed user question
- Find top-K similar chunks
- Return with metadata + scores

### Stage 6: Answer Generation
- Inject retrieved context
- Use Gemini with grounding prompt
- Enforce source citations
- Calculate confidence

### Stage 7: API Delivery
- Clean REST endpoints
- Structured responses
- Error handling
- Health checks

## Testing

### Using Postman

1. **Import Collection**
   - Use `postman_collection.json` (to be created)

2. **Test Upload**
   ```
   POST http://localhost:8000/api/v1/upload
   Body: form-data → file: your-document.pdf
   ```

3. **Test Q&A**
   ```
   POST http://localhost:8000/api/v1/ask
   Body: JSON
   {
     "question": "What is the main topic?",
     "top_k": 5
   }
   ```

### Using cURL

```bash
# Upload
curl -X POST -F "file=@test.pdf" http://localhost:8000/api/v1/upload

# Ask
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is this about?"}' \
  http://localhost:8000/api/v1/ask

# Health
curl http://localhost:8000/api/v1/health
```

## Deployment

### Local Testing
```bash
python -m app.main
```

### Production Deployment (Render)

1. Push to GitHub
2. Connect repo to Render
3. Set environment variables
4. Deploy

```yaml
# render.yaml
services:
  - type: web
    name: researchmind-ai
    env: python
    plan: standard
    startCommand: "uvicorn app.main:app --host 0.0.0.0 --port 8000"
    envVars:
      - key: GEMINI_API_KEY
        scope: run
      - key: QDRANT_URL
        scope: run
```

## Best Practices

### When Answering Questions
1. ✅ Only use retrieved context
2. ✅ Cite sources explicitly
3. ✅ Be concise and factual
4. ✅ Return confidence scores
5. ❌ Don't hallucinate
6. ❌ Don't ignore context

### For Chunking
- Overlap is critical (maintains context boundaries)
- Semantic splitting > fixed splitting
- Page markers help with citations

### For Embeddings
- Dense embeddings > sparse
- Semantic-transformers good default
- Dimension tradeoff: 384 vs 1536

### For Retrieval
- Top-K=5 usually sufficient
- Monitor similarity scores
- Filter by document if needed

## Debugging

### Check Vector DB
```python
# Get collection stats
vector_store.get_collection_info()
```

### Check Embeddings
```python
# Verify embedding dimension
embeddings.get_embedding_dimension()
```

### Monitor Logs
```bash
# Watch logs (in production)
tail -f logs/app.log
```

## Interview Talking Points

1. **Why RAG?** - Reduces hallucination, enables citations, stays up-to-date
2. **Chunking Strategy** - Overlap preserves meaning at chunk boundaries
3. **Embeddings** - Semantic vectors enable similarity search
4. **Qdrant** - Production-ready, local-first, excellent performance
5. **Citations** - Store metadata, return with results
6. **Confidence** - Based on retrieval quality and source count

## Future Enhancements

### Short-term (This Month)
- [ ] LangGraph workflow
- [ ] Streaming responses
- [ ] Basic memory

### Medium-term (Q1)
- [ ] Database persistence
- [ ] Redis caching
- [ ] Advanced metrics

### Long-term (Q2+)
- [ ] Multi-document comparison
- [ ] Advanced topic extraction
- [ ] Quiz agent
- [ ] Research notes agent

## License

MIT

## Author

Built with professional backend engineering practices.
