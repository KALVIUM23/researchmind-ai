"""Answer Generation Service - Stage 6"""

import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class AnswerGenerationService:
    """Generate grounded answers using Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize answer generation service
        
        Args:
            api_key: Google Gemini API key
            model_name: Model to use for generation
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            logger.info(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            raise
    
    def generate_answer(self, prompt: str, temperature: float = 0.7, 
                       max_tokens: int = 1024) -> str:
        """
        Generate answer from prompt
        
        Args:
            prompt: Formatted prompt with context and question
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            
        Returns:
            Generated answer
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            answer = response.text
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def extract_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source citations from retrieved chunks
        
        Args:
            answer: Generated answer
            chunks: Retrieved context chunks
            
        Returns:
            List of citations
        """
        citations = []
        for chunk in chunks:
            citations.append({
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "document_id": chunk["metadata"]["document_id"],
                "similarity_score": chunk["similarity_score"]
            })
        
        # Remove duplicates
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation["source"], citation["page"])
            if key not in seen:
                unique_citations.append(citation)
                seen.add(key)
        
        return unique_citations
    
    def calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieval quality
        
        Args:
            chunks: Retrieved context chunks
            
        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Average similarity score
        avg_score = sum(chunk["similarity_score"] for chunk in chunks) / len(chunks)
        
        # Consider number of sources
        sources = len(set(chunk["metadata"]["source"] for chunk in chunks))
        source_bonus = min(0.2, sources * 0.1)
        
        confidence = min(1.0, avg_score + source_bonus)
        return confidence
    
    def generate_grounded_answer(self, question: str, context_prompt: str, 
                                chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Generate complete grounded answer with citations
        
        Args:
            question: User's question
            context_prompt: Formatted prompt with context
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (answer, citations, confidence)
        """
        try:
            # Generate answer
            answer = self.generate_answer(context_prompt)
            
            # Extract citations
            citations = self.extract_citations(answer, chunks)
            
            # Calculate confidence
            confidence = self.calculate_confidence(chunks)
            
            return answer, citations, confidence
            
        except Exception as e:
            logger.error(f"Error generating grounded answer: {str(e)}")
            raise
