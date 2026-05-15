# Deployment & Setup Guide

## Local Development Setup

### Prerequisites
- Python 3.9+
- Qdrant (local instance)
- Google Gemini API key

### Step 1: Clone Repository
```bash
cd c:\researchmind-ai
git clone https://github.com/YOUR_USERNAME/researchmind-ai.git
cd backend
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy template
copy .env .env.local

# Edit with your values
# - GEMINI_API_KEY: Get from Google AI Studio
# - QDRANT_URL: Local instance URL (http://localhost:6333)
```

### Step 5: Start Qdrant (if local)
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest

# Or download from https://github.com/qdrant/qdrant
```

### Step 6: Run FastAPI Server
```bash
python -m app.main

# Server runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

## API Testing

### Using Postman
1. Open Postman
2. Import `postman_collection.json`
3. Set `BASE_URL` variable to `http://localhost:8000`
4. Test endpoints

### Using cURL

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**Upload PDF:**
```bash
curl -X POST -F "file=@sample.pdf" http://localhost:8000/api/v1/upload
```

**Ask Question:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":5}' \
  http://localhost:8000/api/v1/ask
```

## Production Deployment (Render)

### Prerequisites
- GitHub account with repository
- Render account
- Qdrant Cloud instance (or self-hosted)

### Step 1: Create GitHub Repository
```bash
# In your GitHub account, create: researchmind-ai
# Then update remote URL:
git remote set-url origin https://github.com/YOUR_USERNAME/researchmind-ai.git
```

### Step 2: Push Code
```bash
git push -u origin feature/milestone-rag-backend-pipeline

# Create PR on GitHub and merge to main
```

### Step 3: Create Render Service
1. Go to https://render.com
2. Connect GitHub account
3. Create new Web Service
4. Select repository: `researchmind-ai`
5. Configure:
   - **Name**: researchmind-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Step 4: Set Environment Variables
In Render dashboard:
```
GEMINI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=research_docs
```

### Step 5: Deploy
- Render auto-deploys on push to main
- Monitor deployment status in dashboard
- Access API at: `https://researchmind-api.onrender.com`

## Troubleshooting

### Qdrant Connection Error
```
ConnectionError: Failed to connect to Qdrant
```
**Solution:**
- Ensure Qdrant is running
- Check QDRANT_URL in .env
- For local: verify Docker is running

### Gemini API Error
```
google.auth.exceptions.DefaultCredentialsError
```
**Solution:**
- Get API key from https://ai.google.dev
- Add to .env: `GEMINI_API_KEY=your_key`
- Restart server

### PDF Upload Error
```
ValueError: PDF file is corrupted
```
**Solution:**
- Use valid PDF files
- Check file size < 50MB
- Try different PDF

### Embedding Error
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Use CPU mode (default)
- Reduce batch size in config
- Check available memory

## Performance Monitoring

### Key Metrics to Monitor
1. **Response Time**: Target < 5s for answer
2. **Retrieval Accuracy**: Top-K similarity > 0.7
3. **Confidence Scores**: Average > 0.75
4. **Vector DB Size**: Total chunks/points
5. **Error Rate**: < 1%

### Logs Location
```
# Development
Console output

# Production (Render)
Dashboard → Service → Logs
```

## Scaling Considerations

### When to Scale
- Retrieval time > 5 seconds
- Vector DB points > 1M
- QPS > 100

### Scaling Strategies
1. **Vector DB**: Move to Qdrant Cloud
2. **Caching**: Add Redis layer
3. **Database**: Add PostgreSQL for metadata
4. **API Servers**: Horizontal scaling on Render
5. **Embeddings**: Use GPU for faster inference

## Security Checklist

- [ ] Never commit .env file
- [ ] Use environment variables for secrets
- [ ] Validate file uploads (PDF only)
- [ ] Rate limit API endpoints
- [ ] Use HTTPS in production
- [ ] Implement CORS properly
- [ ] Add API authentication (future)
- [ ] Encrypt sensitive data at rest

## Monitoring & Logging

### Set Up Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Useful Log Points
- PDF upload start/end
- Chunk creation count
- Embedding generation time
- Vector search scores
- Answer generation time

## Database Migration Plan

Future: Move from in-memory to PostgreSQL

```python
# Document metadata table
documents = Table(
    'documents',
    Column('id', String, primary_key=True),
    Column('filename', String),
    Column('pages', Integer),
    Column('chunks_count', Integer),
    Column('created_at', DateTime),
    Column('status', String)
)

# Chunk metadata table
chunks = Table(
    'chunks',
    Column('id', String, primary_key=True),
    Column('document_id', String, ForeignKey('documents.id')),
    Column('text', String),
    Column('page', Integer),
    Column('chunk_index', Integer)
)
```

## Next Steps

1. **This Week**
   - Deploy to Render
   - Test all endpoints
   - Get Qdrant instance

2. **Next Week**
   - Add streaming responses
   - Implement chat memory
   - Create admin dashboard

3. **Month 2**
   - LangGraph agents
   - Document comparison
   - Quiz generation

