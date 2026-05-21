"""Upload API Routes"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from datetime import datetime
import logging
import os
from pathlib import Path

from ..services.parser_service import ParserService
from ..utils.file_handler import generate_document_id, sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["documents"])

# Configuration
MAX_FILE_SIZE = 52428800  # 50MB
ALLOWED_EXTENSIONS = {'.pdf'}
UPLOAD_DIR = "uploads"


async def get_document_service():
    """Dependency injection placeholder - overridden in main.py"""
    return None


async def get_document_store():
    """Dependency injection placeholder - overridden in main.py"""
    return None


async def get_vector_store():
    """Dependency injection placeholder - overridden in main.py"""
    return None


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_service=Depends(get_document_service)
):
    """
    Upload and process PDF document
    
    Complete pipeline:
    1. Validate PDF format and size
    2. Extract text and metadata
    3. Generate chunks
    4. Create embeddings
    5. Store in vector database
    6. Save document metadata
    
    Returns:
        UploadResponse with document status and processing results
    """
    parser = ParserService()
    file_size = 0
    file_path = None
    sanitized_filename = None
    
    try:
        # Step 1: Validate file type
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type attempted: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )
        
        # Step 2: Read and validate file size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file.filename} ({file_size} bytes)")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
        
        if file_size == 0:
            logger.warning(f"Empty file uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        logger.info(f"Starting upload process for {file.filename} ({file_size} bytes)")
        
        # Step 3: Save file to disk
        Path(UPLOAD_DIR).mkdir(exist_ok=True)
        sanitized_filename = sanitize_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"File saved to {file_path}")
        
        # Step 4: Validate PDF integrity
        is_valid, error_msg = parser.validate_pdf(file_path)
        if not is_valid:
            logger.error(f"Invalid PDF: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid PDF file: {error_msg}"
            )
        
        # Step 5: Extract metadata
        logger.info("Extracting PDF metadata...")
        metadata = parser.extract_metadata(file_path)
        
        # Step 6: Generate document ID
        document_id = generate_document_id()
        logger.info(f"Generated document ID: {document_id}")
        
        # Step 7: Process PDF through complete pipeline
        logger.info("Starting document processing pipeline...")
        result = await document_service.process_pdf(
            file_path=file_path,
            filename=file.filename,
            document_id=document_id
        )
        
        logger.info(f"Successfully processed {file.filename}: {result['chunks']} chunks")
        
        # Step 8: Return success response
        response = {
            "document_id": result["document_id"],
            "filename": result["filename"],
            "pages": result["pages"],
            "chunks_created": result["chunks"],
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "status": result["status"],
            "pdf_metadata": {
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "subject": metadata.get("subject"),
            },
            "timestamp": datetime.now().isoformat(),
            "message": f"Successfully indexed {result['chunks']} chunks from {result['pages']} pages"
        }
        
        logger.info(f"Upload complete: {response}")
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        
        # Cleanup on error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up failed upload: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file: {cleanup_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@router.get("/documents")
async def list_documents(document_store=Depends(get_document_store)):
    """
    List all uploaded documents
    
    Returns:
        List of documents with metadata
    """
    try:
        if document_store is None:
            raise HTTPException(
                status_code=503,
                detail="Document service not available"
            )
        
        documents = document_store.list_documents()
        return {
            "total": len(documents),
            "documents": documents,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    document_store=Depends(get_document_store)
):
    """Get document metadata by ID"""
    try:
        if document_store is None:
            raise HTTPException(
                status_code=503,
                detail="Document service not available"
            )
        
        doc = document_store.get_document(document_id)
        if not doc:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document": doc,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    document_store=Depends(get_document_store),
    vector_store=Depends(get_vector_store)
):
    """Delete document and all its chunks from vector database"""
    try:
        if document_store is None or vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Document service not available"
            )
        
        # Check if document exists
        doc = document_store.get_document(document_id)
        if not doc:
            logger.warning(f"Attempted to delete non-existent document: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleting document: {document_id}")
        
        # Delete from vector store
        vector_store.delete_by_document(document_id)
        
        # Delete from document store
        document_store.delete_document(document_id)
        
        logger.info(f"Successfully deleted document: {document_id}")
        
        return {
            "status": "deleted",
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Document {document_id} and all its chunks have been deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
