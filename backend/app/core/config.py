"""
Centralized Configuration Management

This module centralizes all application configuration and settings.
All environment variables are loaded here to ensure:
- No hardcoded secrets
- Support for environment switching (dev, staging, prod)
- Type-safe configuration access
- Easy deployment across environments
"""

from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file - try multiple paths
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # From backend/app/core/config.py -> backend/.env
    Path.cwd() / ".env",  # Current working directory
    Path.cwd() / "backend" / ".env",  # If running from project root
]

for env_file in env_paths:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        break


class Settings:
    """
    Application Settings
    
    All configuration is centralized here.
    Environment variables override defaults.
    """
    
    def __init__(self):
        # Application
        self.app_name = os.getenv("APP_NAME", "ResearchMind AI")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # API Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        
        # Gemini API - allow empty key for deployment
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Qdrant Vector Database
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "researchmind")
        
        # Text Processing
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # File Upload
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB in bytes
        self.upload_directory = os.getenv("UPLOAD_DIRECTORY", "uploads")
        
        # Embeddings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        
        # Retrieval
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Returns:
        Settings: Singleton instance of application settings
    
    Usage:
        from app.core.config import get_settings
        settings = get_settings()
        api_key = settings.gemini_api_key
    """
    return Settings()


# Create default instance for module imports
settings = get_settings()
