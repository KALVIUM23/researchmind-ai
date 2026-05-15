"""Upload API Routes"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
import logging
import os
import shutil

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), document_service=None):
    """
    Upload and process PDF document
    
    Returns:
        UploadResponse with document status
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Validate file size
        file_content = await file.read()
        file_size = len(file_content)
        max_size = 52428800  # 50MB
        
        if file_size > max_size:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {max_size} bytes")
        
        # Save file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Generate document ID
        from app.utils.file_handler import generate_document_id
        document_id = generate_document_id()
        
        # Process PDF
        result = await document_service.process_pdf(file_path, file.filename, document_id)
        
        logger.info(f"Successfully uploaded: {file.filename} ({file_size} bytes)")
        
        return {
            "filename": result["filename"],
            "pages": result["pages"],
            "chunks": result["chunks"],
            "status": result["status"],
            "document_id": result["document_id"],
            "file_size": file_size,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/documents")
async def list_documents(document_store=None):
    """
    List all uploaded documents
    
    Returns:
        List of documents
    """
    try:
        documents = document_store.list_documents()
        return {
            "total": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str, document_store=None):
    """Get document metadata"""
    try:
        doc = document_store.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, document_store=None, vector_store=None):
    """Delete document and its chunks"""
    try:
        vector_store.delete_by_document(document_id)
        document_store.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
