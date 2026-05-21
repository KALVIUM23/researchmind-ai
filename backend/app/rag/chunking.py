"""Semantic Chunking Pipeline - Stage 2"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from .chunk_management import (
    EnhancedChunkMetadata,
    ChunkIdentifier,
    ChunkPositionTracker,
    ChunkQualityAnalyzer,
    ChunkDeduplicationManager,
)

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata container for chunks - DEPRECATED, use EnhancedChunkMetadata"""
    def __init__(self, source: str, page: int, chunk_index: int, document_id: str):
        self.source = source
        self.page = page
        self.chunk_index = chunk_index
        self.document_id = document_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id
        }


class ChunkingService:
    """Handle semantic text chunking with enhanced metadata and tracking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunking service
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        # Initialize helper services
        self.quality_analyzer = ChunkQualityAnalyzer()
        self.deduplication_manager = ChunkDeduplicationManager()
    
    def chunk_text(self, text: str, source: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks with enhanced metadata
        
        Args:
            text: Full document text
            source: Source filename
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata and content
        """
        try:
            logger.info(f"Starting chunking process for {source}")
            
            # Initialize position tracker for accurate positions
            position_tracker = ChunkPositionTracker(text)
            
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            chunked_data = []
            duplicate_count = 0
            
            for chunk_index, chunk in enumerate(chunks):
                # Generate unique chunk ID
                chunk_id = ChunkIdentifier.generate_chunk_id()
                chunk_hash = ChunkIdentifier.generate_chunk_hash(text, document_id, chunk_index)
                
                # Check for duplicates
                is_unique = self.deduplication_manager.register_chunk(chunk_id, chunk_hash)
                if not is_unique:
                    duplicate_count += 1
                    logger.debug(f"Duplicate chunk detected at index {chunk_index}")
                    continue
                
                # Extract page number from chunk
                page_num = self._extract_page_number(chunk)
                
                # Track character positions
                char_start, char_end = position_tracker.record_chunk(chunk)
                
                # Analyze chunk quality
                quality_metrics = self.quality_analyzer.analyze_chunk(chunk)
                
                # Create enhanced metadata
                metadata = EnhancedChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                    text=chunk,
                    char_start=char_start,
                    char_end=char_end,
                )
                
                chunked_data.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "metadata": metadata.to_dict(),
                    "quality_metrics": quality_metrics,
                    "vector_store_payload": metadata.to_vector_store_payload(),
                })
            
            logger.info(
                f"Successfully created {len(chunked_data)} unique chunks from {source} "
                f"({duplicate_count} duplicates filtered)"
            )
            
            return chunked_data
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}", exc_info=True)
            raise
    
    def chunk_text_with_context(self, text: str, source: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with surrounding context for better retrieval
        
        Args:
            text: Full document text
            source: Source filename
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries with context information
        """
        try:
            logger.info(f"Chunking with context for {source}")
            
            position_tracker = ChunkPositionTracker(text)
            chunks = self.splitter.split_text(text)
            
            chunked_data = []
            
            for chunk_index, chunk in enumerate(chunks):
                chunk_id = ChunkIdentifier.generate_chunk_id()
                page_num = self._extract_page_number(chunk)
                char_start, char_end = position_tracker.record_chunk(chunk)
                
                # Get surrounding context
                context = position_tracker.get_text_around_chunk(char_start, char_end, context_chars=100)
                
                metadata = EnhancedChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                    text=chunk,
                    char_start=char_start,
                    char_end=char_end,
                )
                
                chunked_data.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "metadata": metadata.to_dict(),
                    "context": {
                        "before": context["before"],
                        "after": context["after"],
                    },
                })
            
            logger.info(f"Created {len(chunked_data)} chunks with context")
            return chunked_data
            
        except Exception as e:
            logger.error(f"Error chunking with context: {str(e)}", exc_info=True)
            raise
    
    def _extract_page_number(self, chunk: str) -> int:
        """Extract page number from chunk"""
        if "[PAGE" in chunk:
            try:
                start = chunk.find("[PAGE") + 5
                end = chunk.find("]", start)
                return int(chunk[start:end].strip())
            except:
                return 0
        return 0
    
    def update_chunk_size(self, chunk_size: int, chunk_overlap: int):
        """Update chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info(f"Updated chunk size to {chunk_size} with overlap {chunk_overlap}")
