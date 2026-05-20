"""
Centralized Configuration Management

This module centralizes all application configuration and settings.
All environment variables are loaded here to ensure:
- No hardcoded secrets
- Support for environment switching (dev, staging, prod)
- Type-safe configuration access
- Easy deployment across environments
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application Settings
    
    All configuration is centralized here.
    Environment variables override defaults.
    """
    
    # Application
    app_name: str = "ResearchMind AI"
    debug: bool = True
    environment: str = "development"  # development, staging, production
    
    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-pro"
    
    # Qdrant Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "researchmind"
    
    # Text Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # File Upload
    max_file_size: int = 52428800  # 50MB in bytes
    upload_directory: str = "uploads"
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval
    retrieval_top_k: int = 5
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        # Use absolute path to .env file
        env_file = str(Path(__file__).parent.parent.parent / ".env")
        case_sensitive = False
        extra = "allow"  # Allow extra fields
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validate critical settings
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
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
