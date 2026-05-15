# Architecture & Technical Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ResearchMind AI Backend                        │
│                                                                           │
│  ┌──────────────┐         ┌──────────────────────────────────────────┐  │
│  │   FastAPI    │         │          Request Processing              │  │
│  │   Endpoints  │────────▶│ - Route Dispatch                         │  │
│  └──────────────┘         │ - Input Validation                       │  │
│        │                  │ - Error Handling                         │  │
│        │                  └──────────────┬───────────────────────────┘  │
│        │                                 │                               │
│        ▼                                 ▼                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              SERVICE LAYER                                      │   │
│  │  ┌────────────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ Document Service       │  │ Answer Service              │  │   │
│  │  │ - PDF Upload           │  │ - Question Processing       │  │   │
│  │  │ - Pipeline Orchestration│ │ - RAG Coordination          │  │   │
│  │  │ - Metadata Management  │  │ - Response Generation       │  │   │
│  │  └────────────────────────┘  └──────────────────────────────┘  │   │
│  └───────────────────────┬────────────────────┬────────────────────┘   │
│                          │                    │                         │
│        ┌─────────────────┼────────────────────┼────────────────────┐   │
│        │                 │                    │                    │   │
│        ▼                 ▼                    ▼                    ▼   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │   PDF        │  │   Chunking   │  │ Embeddings  │  │  Retrieval   │ │
│  │  Ingestion   │  │   Service    │  │  Service    │  │   Service    │ │
│  │  (Stage 1)   │  │  (Stage 2)   │  │ (Stage 3)   │  │  (Stage 5)   │ │
│  │              │  │              │  │             │  │              │ │
│  │ PyPDF2       │  │ LangChain    │  │ sentence-   │  │ Cosine       │ │
│  │ Text Ext     │  │ Splitter     │  │ transformers │ │ Similarity   │ │
│  │              │  │ Overlap:200  │  │ 384-dim     │  │ Top-K        │ │
│  │              │  │ Size:1000    │  │             │  │              │ │
│  └──────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
│        │                 │                    │                    │   │
│        └─────────────────┼────────────────────┼────────────────────┘   │
│                          │                    │                        │
│                          ▼                    ▼                        │
│              ┌────────────────────────────────────────┐                │
│              │     Vector Store (Qdrant)             │                │
│              │     (Stage 4)                         │                │
│              │                                        │                │
│              │ - Collection: research_docs            │                │
│              │ - Distance: Cosine                     │                │
│              │ - Dim: 384                             │                │
│              │ - Payload: {text, source, page, ...}  │                │
│              └────────────────────────────────────────┘                │
│                                    │                                    │
│                                    ▼                                    │
│              ┌────────────────────────────────────────┐                │
│              │  Answer Generation                     │                │
│              │  (Stage 6)                             │                │
│              │                                        │                │
│              │ - Context Injection                    │                │
│              │ - Gemini API Call                      │                │
│              │ - Grounding Prompt                     │                │
│              │ - Citation Extraction                  │                │
│              │ - Confidence Calculation               │                │
│              └────────────────────────────────────────┘                │
│                          │                                              │
│                          ▼                                              │
│              ┌────────────────────────────────────────┐                │
│              │   Response Formatting                  │                │
│              │   & Delivery                           │                │
│              │                                        │                │
│              │ {answer, citations, confidence,        │                │
│              │  retrieval_stats, chunks_count}        │                │
│              └────────────────────────────────────────┘                │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     ▼
                            ┌────────────────────┐
                            │   Client/Frontend  │
                            │   HTTP Response    │
                            └────────────────────┘
```

## Data Flow

### PDF Upload Flow
```
1. Client uploads PDF
   ↓
2. Validate file (PDF, < 50MB)
   ↓
3. Save temporarily
   ↓
4. Extract text with page markers
   ↓
5. Generate document_id
   ↓
6. Chunk text (1000 chars, 200 overlap)
   ↓
7. Generate embeddings (384-dim)
   ↓
8. Store in Qdrant with metadata
   ↓
9. Save document metadata
   ↓
10. Return response {filename, pages, chunks, document_id}
```

### Question Answering Flow
```
1. Client asks question
   ↓
2. Validate question (not empty)
   ↓
3. Generate question embedding
   ↓
4. Search Qdrant (top-5 similar chunks)
   ↓
5. Format context with citations
   ↓
6. Call Gemini with context + question
   ↓
7. Extract answer text
   ↓
8. Calculate confidence score
   ↓
9. Return {answer, citations, confidence, stats}
```

## Key Design Decisions

### 1. Semantic Chunking
- **Why**: Preserves context at chunk boundaries
- **How**: RecursiveCharacterTextSplitter with 200-char overlap
- **Impact**: Better retrieval accuracy, more nuanced answers

### 2. Dense Embeddings
- **Why**: Semantic search requires dense vectors
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Trade-off**: Speed vs. quality (SBERT is fast and accurate)

### 3. Qdrant Vector DB
- **Why**: Production-ready, local-first, excellent filtering
- **Alternative**: Pinecone (cloud), FAISS (lightweight)
- **Use Case**: Our use case fits perfectly: multi-tenant, metadata filtering

### 4. Gemini LLM
- **Why**: Excellent context window, good instruction-following
- **Grounding Prompt**: Forces model to cite sources and avoid hallucination
- **Alternative**: GPT-4, Claude (cost trade-off)

### 5. FastAPI Framework
- **Why**: Async-first, automatic validation, OpenAPI docs
- **Alternative**: Flask (simpler), Django (more features)
- **Scalability**: Built for async I/O, perfect for RAG workloads

## Storage Architecture

### Vector Store (Qdrant)
```json
{
  "collection": "research_docs",
  "points": [
    {
      "id": "hash(text)",
      "vector": [0.123, -0.456, ..., 0.789],  // 384 dims
      "payload": {
        "text": "Document text chunk...",
        "source": "document.pdf",
        "page": 4,
        "chunk_index": 12,
        "document_id": "uuid-string"
      }
    }
  ]
}
```

### Document Store (In-Memory, Future: PostgreSQL)
```json
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "pages": 10,
  "chunks_count": 45,
  "file_size": 2048576,
  "status": "indexed",
  "created_at": "2024-01-15T10:30:00"
}
```

## Performance Characteristics

### Latency Breakdown (Typical Query)
- Question embedding: 100ms
- Vector search (top-5): 50ms
- Context formatting: 10ms
- Gemini API call: 2000-3000ms
- Response formatting: 5ms
- **Total: 2.2-3.2 seconds**

### Throughput
- **Sequential**: 1-2 questions/second per instance
- **Concurrent**: Limited by Gemini API quota
- **Scaling**: Horizontal via Render auto-scaling

### Vector DB Efficiency
- Collection size: 150 points = ~500KB
- Query cost: O(log N) with indexing
- Memory per point: ~1.5KB (384-dim float32 + payload)

## Scalability Strategy

### Phase 1: Current (Single Server)
- 1 FastAPI instance
- Local Qdrant
- In-memory document store
- **Capacity**: ~100K documents

### Phase 2: Production (Week 2)
- Render auto-scaling (2-4 instances)
- Qdrant Cloud
- PostgreSQL for metadata
- Redis for caching
- **Capacity**: ~1M documents

### Phase 3: Enterprise (Month 2+)
- Kubernetes (if needed)
- Distributed vector search
- Multi-region deployment
- Advanced caching strategy
- **Capacity**: Unlimited

## Security Architecture

### Input Validation
```python
# PDF files only
- File extension check: .pdf
- MIME type validation
- Size limit: 50MB

# Questions
- Non-empty string
- Max length: 2000 chars
- No malicious patterns
```

### API Security
- CORS configured
- Rate limiting (future)
- Input sanitization
- Error message obfuscation

### Secrets Management
- Environment variables (never hardcode)
- .env files (never commit)
- Production: Render secret manager

## Integration Points

### External Services
1. **Google Gemini API**
   - Endpoint: generativeai library
   - Quota: Based on plan
   - Fallback: None (critical path)

2. **Qdrant Vector DB**
   - Protocol: HTTP/gRPC
   - Auth: API key (optional for local)
   - Fallback: FAISS (local, slower)

### Future Integrations
- PostgreSQL (persistence)
- Redis (caching)
- LangGraph (agents)
- Stripe (payments, if SaaS)

## Error Handling Strategy

### Retrieval Failures
```python
if not chunks:
    return {
        "answer": "No relevant information found",
        "confidence": 0.0
    }
```

### LLM Failures
```python
try:
    answer = generate_answer(prompt)
except Exception:
    logger.error(...)
    raise HTTPException(500, "Failed to generate answer")
```

### Database Failures
```python
try:
    results = vector_store.search(embedding)
except ConnectionError:
    # Could fallback to FAISS index
    raise HTTPException(503, "Vector DB unavailable")
```

## Monitoring & Observability

### Key Metrics
- API response time (p95, p99)
- Vector search latency
- Gemini API latency
- Error rate
- Cache hit rate
- Document count
- Total embeddings stored

### Logging Strategy
- Info: Major operations
- Warning: Degradation
- Error: Failures
- Debug: Detailed flow (dev only)

### Health Checks
- Vector DB connectivity
- Gemini API availability
- File system access
- Memory usage

## Testing Strategy

### Unit Tests
- PDF extraction
- Text chunking
- Embedding generation
- Cosine similarity
- Citation extraction

### Integration Tests
- Full upload pipeline
- Full Q&A pipeline
- Multi-document queries
- Error scenarios

### Load Tests
- Concurrent uploads
- Concurrent queries
- Vector DB scaling
- API rate limiting

## Compliance & Governance

### Data Privacy
- PDFs stored locally (configurable)
- No persistent chat history
- Document deletion support
- GDPR compliance (future)

### Audit Trail
- All uploads logged
- Question logs (anonymized)
- Error tracking
- API usage metrics

## Deployment Architecture

### Development
```
Local Machine
├── FastAPI (port 8000)
├── Qdrant (Docker, port 6333)
└── Environment: .env
```

### Production (Render)
```
Render.com
├── Web Service: FastAPI
├── Environment: Secret manager
└── External: Qdrant Cloud
```

## What Recruiters Should Know

### 1. Semantic Understanding
- Why embeddings matter
- How vector similarity works
- Why dense vs sparse

### 2. Grounding Strategy
- How citations prevent hallucination
- Why context injection is critical
- How confidence scoring works

### 3. System Design
- Trade-offs: Chunking size, overlap, embedding model
- Scalability decisions
- Performance optimization

### 4. Production Readiness
- Error handling patterns
- Logging strategy
- Deployment automation
- Monitoring approach

