"""Document Management Service"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class DocumentStore:
    """In-memory document metadata store (later move to DB)"""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
    
    def save_document(self, document_id: str, metadata: Dict[str, Any]):
        """Save document metadata"""
        self.documents[document_id] = {
            **metadata,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Saved metadata for document: {document_id}")
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata"""
        return self.documents.get(document_id)
    
    def list_documents(self) -> list:
        """List all documents"""
        return list(self.documents.values())
    
    def delete_document(self, document_id: str):
        """Delete document metadata"""
        if document_id in self.documents:
            del self.documents[document_id]
            logger.info(f"Deleted metadata for document: {document_id}")


class DocumentService:
    """Orchestrate document processing pipeline"""
    
    def __init__(self, ingestion_service, chunking_service, embeddings_service, 
                 vector_store, document_store):
        """Initialize document service"""
        self.ingestion = ingestion_service
        self.chunking = chunking_service
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.doc_store = document_store
    
    async def process_pdf(self, file_path: str, filename: str, document_id: str) -> Dict[str, Any]:
        """
        Process PDF end-to-end
        
        Args:
            file_path: Path to PDF file
            filename: Original filename
            document_id: Document ID
            
        Returns:
            Processing result
        """
        try:
            # Stage 1: Extract text
            logger.info(f"Stage 1: Extracting text from {filename}")
            full_text, page_count = self.ingestion.extract_text_from_pdf(file_path)
            
            # Stage 2: Chunk text
            logger.info("Stage 2: Chunking text")
            chunks = self.chunking.chunk_text(full_text, filename, document_id)
            
            # Stage 3: Generate embeddings
            logger.info("Stage 3: Generating embeddings")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embeddings.embed_texts(chunk_texts)
            
            # Stage 4: Store in vector DB
            logger.info("Stage 4: Storing in vector DB")
            point_ids = self.vector_store.add_chunks(chunks, embeddings)
            
            # Save document metadata
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "pages": page_count,
                "chunks_count": len(chunks),
                "file_size": self._get_file_size(file_path),
                "status": "indexed"
            }
            self.doc_store.save_document(document_id, metadata)
            
            logger.info(f"Successfully processed {filename}")
            return {
                "document_id": document_id,
                "filename": filename,
                "pages": page_count,
                "chunks": len(chunks),
                "status": "indexed"
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        import os
        return os.path.getsize(file_path)
