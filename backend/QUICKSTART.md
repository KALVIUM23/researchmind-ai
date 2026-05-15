# Quick Start Guide

## 5-Minute Setup

### Step 1: Clone & Setup (2 min)
```bash
cd c:\researchmind-ai\backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure (1 min)
```bash
# Edit .env with your keys
# - Get GEMINI_API_KEY from https://ai.google.dev
# - Set QDRANT_URL to http://localhost:6333
```

### Step 3: Start Services (1 min)
```bash
# Terminal 1: Start Qdrant (Docker required)
docker run -p 6333:6333 qdrant/qdrant:latest

# Terminal 2: Start FastAPI
python -m app.main
```

### Step 4: Test (1 min)
```bash
# In new terminal
curl http://localhost:8000/api/v1/health
```

## Testing the API

### Upload a PDF
```bash
curl -X POST -F "file=@sample.pdf" http://localhost:8000/api/v1/upload
```

### Ask a Question
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}' \
  http://localhost:8000/api/v1/ask
```

### Check Health
```bash
curl http://localhost:8000/api/v1/health
```

## API Documentation

**Interactive Docs**: http://localhost:8000/docs

- Swagger UI for testing all endpoints
- Try-it-out feature
- Schema validation

## Project Structure

```
backend/
├── app/
│   ├── api/              # Routes
│   ├── rag/              # Pipeline
│   ├── services/         # Business logic
│   ├── vectorstore/      # Qdrant
│   ├── models/           # Schemas
│   ├── utils/            # Helpers
│   └── main.py          # Entry point
├── tests/                # Test suite
├── uploads/              # PDF storage
├── README.md            # Full docs
├── DEPLOYMENT.md        # Deploy guide
├── ARCHITECTURE.md      # Technical details
└── requirements.txt     # Dependencies
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/api/v1/health` | GET | Status check |
| `/api/v1/upload` | POST | Upload PDF |
| `/api/v1/ask` | POST | Ask question |
| `/api/v1/documents` | GET | List docs |
| `/docs` | GET | Interactive API docs |

## Common Tasks

### Change Chunk Size
Edit `app/config.py`:
```python
chunk_size = 1500  # Increase
chunk_overlap = 300  # Increase
```

### Change Embedding Model
Edit `.env`:
```
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Use Different LLM
Edit `app/rag/answer_generation.py`:
```python
# Change from Gemini to another LLM
model = genai.GenerativeModel("gemini-1.5-pro")
```

## Troubleshooting

**Module not found**: Ensure virtual environment is activated
```bash
venv\Scripts\activate
```

**Port already in use**: Change port in `app/config.py`
```python
port = 8001  # Instead of 8000
```

**Qdrant connection error**: Ensure Docker container running
```bash
docker ps  # Check if running
```

**Gemini API error**: Verify API key in `.env`
```bash
# Test key
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY')"
```

## Next Steps

1. **Deploy to Production**: See `DEPLOYMENT.md`
2. **Understand Architecture**: See `ARCHITECTURE.md`
3. **Learn More**: See `README.md`
4. **Contributing**: Add features, file PRs

## Support

- **Issues**: GitHub issues
- **Docs**: See README.md and ARCHITECTURE.md
- **API Docs**: http://localhost:8000/docs

---

**Ready?** Let's build something great! 🚀
