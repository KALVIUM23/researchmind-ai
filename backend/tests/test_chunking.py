"""Unit tests for chunking service"""

from app.rag.chunking import ChunkingService


def test_chunking_service_initialization():
    """Test chunking service initialization"""
    service = ChunkingService(chunk_size=500, chunk_overlap=100)
    assert service.chunk_size == 500
    assert service.chunk_overlap == 100


def test_chunk_text():
    """Test text chunking"""
    service = ChunkingService(chunk_size=100, chunk_overlap=20)
    
    text = """
    This is a test document with multiple paragraphs.
    
    The second paragraph contains important information that should be preserved
    across chunks due to the overlap setting.
    
    Additional content here to ensure we have enough text to create multiple chunks.
    """
    
    chunks = service.chunk_text(text, "test.pdf", "doc-123")
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
