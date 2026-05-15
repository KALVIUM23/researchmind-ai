from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class UploadResponse(BaseModel):
    """Response model for PDF upload"""
    filename: str
    pages: int
    status: str
    document_id: str
    timestamp: datetime
    file_size: int


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk"""
    source: str
    page: int
    chunk_index: int
    document_id: str


class RetrievedChunk(BaseModel):
    """Retrieved chunk with metadata and relevance score"""
    text: str
    metadata: ChunkMetadata
    similarity_score: float


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="The user's question about the documents")
    document_id: Optional[str] = Field(None, description="Specific document to search in")
    top_k: int = Field(5, description="Number of chunks to retrieve")


class AnswerResponse(BaseModel):
    """Response model for generated answer"""
    answer: str
    citations: List[ChunkMetadata]
    retrieved_chunks: List[RetrievedChunk]
    confidence: float


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents"""
    document_id: str
    filename: str
    pages: int
    upload_time: datetime
    file_size: int
    status: str = "indexed"


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    vectordb_connected: bool
    embeddings_loaded: bool
    timestamp: datetime
