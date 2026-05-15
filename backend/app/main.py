"""FastAPI Main Application - Stage 7: Professional API Design"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.config import get_settings, Settings
from app.rag.ingestion import PDFIngestionService
from app.rag.chunking import ChunkingService
from app.rag.embeddings import EmbeddingsService
from app.rag.retrieval import RetrievalService
from app.rag.answer_generation import AnswerGenerationService
from app.vectorstore.qdrant_store import VectorStoreService
from app.services.document_service import DocumentService, DocumentStore
from app.services.answer_service import AnswerService
from app.api import documents, questions
from app.utils.logger import log_upload

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle context manager for app startup/shutdown
    """
    # Startup
    try:
        settings = get_settings()
        logger.info("Initializing services...")
        
        # Initialize core services
        ingestion_service = PDFIngestionService()
        
        chunking_service = ChunkingService(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        embeddings_service = EmbeddingsService(
            model_name=settings.embedding_model
        )
        
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            embedding_dim=embeddings_service.get_embedding_dimension()
        )
        
        retrieval_service = RetrievalService(
            embeddings_service=embeddings_service,
            vector_store=vector_store
        )
        
        answer_generation_service = AnswerGenerationService(
            api_key=settings.gemini_api_key
        )
        
        document_store = DocumentStore()
        
        document_service = DocumentService(
            ingestion_service=ingestion_service,
            chunking_service=chunking_service,
            embeddings_service=embeddings_service,
            vector_store=vector_store,
            document_store=document_store
        )
        
        answer_service = AnswerService(
            retrieval_service=retrieval_service,
            answer_generation_service=answer_generation_service
        )
        
        # Store services in global dict
        services.update({
            "ingestion": ingestion_service,
            "chunking": chunking_service,
            "embeddings": embeddings_service,
            "vector_store": vector_store,
            "retrieval": retrieval_service,
            "answer_generation": answer_generation_service,
            "document_store": document_store,
            "document_service": document_service,
            "answer_service": answer_service,
        })
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="ResearchMind AI - Document Intelligence Pipeline",
        description="Production-grade RAG backend for document analysis",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency injection function
    def get_services():
        return services
    
    # Include routers with dependency injection
    async def get_document_service():
        return services.get("document_service")
    
    async def get_document_store():
        return services.get("document_store")
    
    async def get_vector_store():
        return services.get("vector_store")
    
    async def get_answer_service():
        return services.get("answer_service")
    
    # Add dependencies to routes
    documents.router.dependency_overrides = {
        "document_service": get_document_service,
        "document_store": get_document_store,
        "vector_store": get_vector_store,
    }
    
    questions.router.dependency_overrides = {
        "answer_service": get_answer_service,
        "vector_store": get_vector_store,
    }
    
    # Register routers
    app.include_router(documents.router)
    app.include_router(questions.router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "ResearchMind AI",
            "version": "1.0.0",
            "description": "Document Intelligence Pipeline with RAG",
            "endpoints": {
                "docs": "/docs",
                "upload": "/api/v1/upload",
                "ask": "/api/v1/ask",
                "documents": "/api/v1/documents",
                "health": "/api/v1/health"
            }
        }
    
    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
