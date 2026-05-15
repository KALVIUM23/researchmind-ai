"""Retrieval Pipeline - Stage 5"""

from typing import List, Dict, Any
from app.rag.embeddings import EmbeddingsService
from app.vectorstore.qdrant_store import VectorStoreService
import logging

logger = logging.getLogger(__name__)


class RetrievalService:
    """Handle document retrieval and context preparation"""
    
    def __init__(self, embeddings_service: EmbeddingsService, vector_store: VectorStoreService):
        """
        Initialize retrieval service
        
        Args:
            embeddings_service: Service for generating embeddings
            vector_store: Vector store service
        """
        self.embeddings = embeddings_service
        self.vector_store = vector_store
    
    def retrieve_context(self, question: str, top_k: int = 5, 
                        document_id: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a question
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            document_id: Optional document filter
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        try:
            # Generate question embedding
            question_embedding = self.embeddings.embed_text(question)
            
            # Search in vector store
            retrieved_chunks = self.vector_store.search(question_embedding, top_k=top_k)
            
            # Optional: Filter by document_id if provided
            if document_id:
                retrieved_chunks = [
                    chunk for chunk in retrieved_chunks
                    if chunk["metadata"]["document_id"] == document_id
                ]
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for question: '{question}'")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {idx}: {chunk['metadata']['source']} | Page {chunk['metadata']['page']}]\n"
                f"{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def prepare_rag_input(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare final RAG input with context and question
        
        Args:
            question: User's question
            chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt for LLM
        """
        context = self.format_context(chunks)
        
        prompt = f"""You are a research assistant. Answer the following question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based only on the provided context
2. If the answer is not in the context, say "The answer is not provided in the available documents"
3. Be concise and factual
4. Always cite the source document and page number

ANSWER:"""
        
        return prompt
    
    def get_retrieval_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about retrieval results"""
        if not chunks:
            return {"total_chunks": 0, "avg_score": 0, "sources": []}
        
        scores = [chunk["similarity_score"] for chunk in chunks]
        sources = set(chunk["metadata"]["source"] for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "sources": list(sources)
        }
