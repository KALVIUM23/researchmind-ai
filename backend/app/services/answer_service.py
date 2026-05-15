"""Answer Service - Orchestrates RAG pipeline"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AnswerService:
    """Orchestrate answer generation with RAG"""
    
    def __init__(self, retrieval_service, answer_generation_service):
        """Initialize answer service"""
        self.retrieval = retrieval_service
        self.answer_gen = answer_generation_service
    
    async def answer_question(self, question: str, top_k: int = 5, 
                             document_id: str = None) -> Dict[str, Any]:
        """
        Answer user question with RAG
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            document_id: Optional document filter
            
        Returns:
            Answer with citations and metadata
        """
        try:
            logger.info(f"Processing question: '{question}'")
            
            # Step 1: Retrieve relevant context
            chunks = self.retrieval.retrieve_context(question, top_k=top_k, 
                                                     document_id=document_id)
            
            if not chunks:
                return {
                    "question": question,
                    "answer": "No relevant information found in the documents.",
                    "citations": [],
                    "confidence": 0.0,
                    "retrieval_stats": {"total_chunks": 0}
                }
            
            # Step 2: Prepare context for LLM
            context_prompt = self.retrieval.prepare_rag_input(question, chunks)
            
            # Step 3: Generate grounded answer
            answer, citations, confidence = self.answer_gen.generate_grounded_answer(
                question, context_prompt, chunks
            )
            
            # Step 4: Get retrieval statistics
            retrieval_stats = self.retrieval.get_retrieval_stats(chunks)
            
            logger.info(f"Generated answer with confidence: {confidence:.2f}")
            
            return {
                "question": question,
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "retrieval_stats": retrieval_stats,
                "retrieved_chunks_count": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
