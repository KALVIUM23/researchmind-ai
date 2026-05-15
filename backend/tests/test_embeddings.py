"""Unit tests for embeddings service"""

from app.rag.embeddings import EmbeddingsService


def test_embeddings_service_initialization():
    """Test embeddings service initialization"""
    service = EmbeddingsService()
    assert service.embedding_dim == 384  # Default model dimension


def test_embed_single_text():
    """Test embedding single text"""
    service = EmbeddingsService()
    
    text = "This is a test sentence for embedding."
    embedding = service.embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    service = EmbeddingsService()
    
    vec1 = [1, 0, 0, 0]
    vec2 = [1, 0, 0, 0]
    
    similarity = service.cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.0001  # Should be 1.0
