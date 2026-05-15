"""Semantic Chunking Pipeline - Stage 2"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata container for chunks"""
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
    """Handle semantic text chunking"""
    
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
    
    def chunk_text(self, text: str, source: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks
        
        Args:
            text: Full document text
            source: Source filename
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            
            chunked_data = []
            for chunk_index, chunk in enumerate(chunks):
                # Extract page number if present
                page_num = self._extract_page_number(chunk)
                
                metadata = ChunkMetadata(
                    source=source,
                    page=page_num,
                    chunk_index=chunk_index,
                    document_id=document_id
                )
                
                chunked_data.append({
                    "text": chunk,
                    "metadata": metadata.to_dict()
                })
            
            logger.info(f"Created {len(chunks)} chunks from {source}")
            return chunked_data
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
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
