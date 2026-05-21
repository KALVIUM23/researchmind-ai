"""Question & Answer API Routes"""

from fastapi import APIRouter, HTTPException, Depends
from ..models.schemas import QuestionRequest, AnswerResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["qa"])


async def get_answer_service():
    """Dependency injection placeholder - overridden in main.py"""
    return None


async def get_vector_store():
    """Dependency injection placeholder - overridden in main.py"""
    return None


@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    answer_service=Depends(get_answer_service)
):
    """
    Ask a question about documents
    
    Args:
        request: QuestionRequest with question
        
    Returns:
        AnswerResponse with answer and citations
    """
    try:
        if answer_service is None:
            raise HTTPException(
                status_code=503,
                detail="Answer service not available"
            )
        
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = await answer_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id
        )
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "retrieval_stats": result["retrieval_stats"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@router.get("/health")
async def health_check(vector_store=Depends(get_vector_store)):
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    try:
        from datetime import datetime
        
        if vector_store is None:
            return {
                "status": "degraded",
                "error": "Vector store not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Check vector store connection
        collection_info = vector_store.get_collection_info()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "vectordb": {
                "connected": True,
                "collection": "research_docs",
                "points_count": collection_info.get("points_count", 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
