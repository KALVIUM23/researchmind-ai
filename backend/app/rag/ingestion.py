"""PDF Ingestion Pipeline - Stage 1"""

from PyPDF2 import PdfReader
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class PDFIngestionService:
    """Handle PDF parsing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
        """
        Extract text and page count from PDF
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (full_text, page_count)
        """
        try:
            reader = PdfReader(file_path)
            pages_count = len(reader.pages)
            
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    # Add page marker for metadata
                    full_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
            
            logger.info(f"Extracted text from {pages_count} pages")
            return full_text, pages_count
            
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
    
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def extract_page_number(text_chunk: str) -> int:
        """Extract page number from chunk if present"""
        if "[PAGE" in text_chunk:
            try:
                start = text_chunk.find("[PAGE") + 5
                end = text_chunk.find("]", start)
                return int(text_chunk[start:end].strip())
            except:
                return 0
        return 0
