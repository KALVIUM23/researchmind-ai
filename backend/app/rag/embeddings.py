"""Embeddings Service - Stage 3"""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
import numpy as np

from .embedding_cache import (
    BatchEmbeddingProcessor,
    MemoryEmbeddingCache,
    DiskEmbeddingCache,
    EmbeddingCache,
)
from .embedding_retry import (
    BatchEmbeddingWithRetry,
    RetryConfig,
    RetryStrategy,
)

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Generate embeddings for text chunks with caching and retry logic"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embeddings service
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return float(dot_product / (norm_v1 * norm_v2))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings"""
        return self.embedding_dim


class OptimizedEmbeddingsService(EmbeddingsService):
    """Enhanced embeddings service with caching and retry logic"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_type: str = "memory",
        use_retry: bool = True,
        retry_config: Optional[RetryConfig] = None,
        batch_size: int = 32,
    ):
        """
        Initialize optimized embeddings service
        
        Args:
            model_name: HuggingFace model name
            use_cache: Enable embedding caching
            cache_type: Type of cache ('memory' or 'disk')
            use_retry: Enable retry logic
            retry_config: Retry configuration
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        
        self.use_cache = use_cache
        self.use_retry = use_retry
        self.batch_size = batch_size
        
        # Initialize cache
        if use_cache:
            if cache_type == "disk":
                self.cache = DiskEmbeddingCache()
            else:
                self.cache = MemoryEmbeddingCache()
            logger.info(f"Initialized {cache_type} embedding cache")
        else:
            self.cache = None
        
        # Initialize batch processor
        self.batch_processor = BatchEmbeddingProcessor(
            embedding_service=self,
            cache=self.cache,
            batch_size=batch_size,
        )
        
        # Initialize retry logic
        if use_retry:
            if retry_config is None:
                retry_config = RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL,
                )
            
            self.batch_retry = BatchEmbeddingWithRetry(
                embedding_service=self,
                retry_config=retry_config,
                batch_size=batch_size,
            )
            logger.info("Initialized retry logic for embeddings")
        else:
            self.batch_retry = None
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Embed multiple chunks with optimization
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            List of chunks with 'embedding' field added
        """
        if self.use_retry and self.batch_retry:
            logger.info("Using batch processing with retry logic")
            return self.batch_retry.process_chunks_with_retry(chunks)
        else:
            logger.info("Using batch processing with cache")
            return self.batch_processor.process_chunks(chunks)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if self.cache and hasattr(self.cache, 'get_stats'):
            return self.cache.get_stats()
        return {}
    
    def get_batch_stats(self) -> dict:
        """Get batch processing statistics"""
        stats = self.batch_processor.get_stats()
        
        if self.batch_retry:
            stats['retry_stats'] = self.batch_retry.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    def reset_stats(self) -> None:
        """Reset all statistics"""
        self.batch_processor.clear_cache()
        if self.batch_retry:
            self.batch_retry.reset_stats()
        logger.info("All stats reset")
