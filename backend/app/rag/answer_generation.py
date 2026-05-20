"""Answer Generation Service - Phase 7 (Grounded LLM Responses with Citations)"""

import google.generativeai as genai
from typing import List, Dict, Any, Tuple, Optional
import logging
from enum import Enum
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Response formatting options"""
    STANDARD = "standard"
    WITH_CITATIONS = "with_citations"
    JSON = "json"
    BULLET_POINTS = "bullet_points"


class AnswerGenerationMetrics:
    """Track answer generation performance"""
    def __init__(self):
        self.total_answers = 0
        self.avg_confidence = 0.0
        self.citations_per_answer = 0
        self.failed_generations = 0
        self.generation_times = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_answers": self.total_answers,
            "avg_confidence": self.avg_confidence,
            "avg_citations": self.citations_per_answer / max(1, self.total_answers),
            "failed_generations": self.failed_generations,
            "avg_generation_time_ms": sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
        }


class AnswerGenerationService:
    """Generate grounded answers using Gemini with advanced features"""
    
    # Grounding prompt templates
    GROUNDING_PROMPTS = {
        "strict": """You MUST answer based ONLY on the provided context. 
If the answer is not in the context, respond: "I cannot answer this question based on the provided documents."
Do not use any external knowledge. Only use information from the context.""",
        
        "balanced": """Answer the question based primarily on the provided context. 
If additional context is needed for clarity, you may use general knowledge but clearly distinguish it.
Always cite source documents and pages for information from context.""",
        
        "lenient": """Use the provided context as the primary source for your answer.
You may supplement with relevant general knowledge when appropriate.
Always indicate which information comes from the provided documents."""
    }
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro", grounding_level: str = "strict"):
        """
        Initialize answer generation service
        
        Args:
            api_key: Google Gemini API key
            model_name: Model to use for generation
            grounding_level: Level of grounding (strict, balanced, lenient)
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.grounding_level = grounding_level
            self.metrics = AnswerGenerationMetrics()
            logger.info(f"[OK] Initialized Gemini model: {model_name}, grounding: {grounding_level}")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            raise
    
    def generate_answer(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1024
    ) -> str:
        """
        Generate answer from prompt with grounding
        
        Args:
            prompt: Formatted prompt with context and question
            temperature: Creativity level (0-1, lower = more focused)
            max_tokens: Maximum response length
            
        Returns:
            Generated answer
        """
        import time
        start_time = time.time()
        
        try:
            # Add grounding instruction
            grounded_prompt = f"{self.GROUNDING_PROMPTS[self.grounding_level]}\n\n{prompt}"
            
            response = self.model.generate_content(
                grounded_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            answer = response.text
            generation_time = (time.time() - start_time) * 1000
            
            self.metrics.total_answers += 1
            self.metrics.generation_times.append(generation_time)
            
            logger.info(f"[OK] Generated answer ({len(answer)} chars, {generation_time:.1f}ms)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            self.metrics.failed_generations += 1
            raise
    
    def extract_citations(
        self, 
        answer: str, 
        chunks: List[Dict[str, Any]],
        dedup_by_page: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract source citations from retrieved chunks
        
        Args:
            answer: Generated answer
            chunks: Retrieved context chunks
            dedup_by_page: Remove duplicate citations from same page
            
        Returns:
            List of citations with metadata
        """
        citations = []
        seen = set()
        
        for chunk in chunks:
            source = chunk["metadata"]["source"]
            page = chunk["metadata"]["page"]
            
            # Deduplication
            if dedup_by_page:
                key = (source, page)
            else:
                key = chunk["id"]
            
            if key not in seen:
                citations.append({
                    "source": source,
                    "page": page,
                    "document_id": chunk["metadata"]["document_id"],
                    "chunk_id": chunk["metadata"].get("chunk_id"),
                    "similarity_score": chunk["similarity_score"],
                    "text_preview": chunk.get("text", "")[:100]
                })
                seen.add(key)
        
        self.metrics.citations_per_answer += len(citations)
        logger.info(f"[OK] Extracted {len(citations)} unique citations")
        return citations
    
    def calculate_confidence(
        self, 
        chunks: List[Dict[str, Any]],
        answer: str,
        min_score_weight: float = 0.7
    ) -> float:
        """
        Calculate confidence score based on multiple factors
        
        Args:
            chunks: Retrieved context chunks
            answer: Generated answer text
            min_score_weight: Weight for minimum similarity score
            
        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Similarity scores (70% weight)
        scores = [chunk["similarity_score"] for chunk in chunks]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        score_confidence = (avg_score * 0.6 + min_score * 0.4) * min_score_weight
        
        # Factor 2: Source diversity (15% weight)
        sources = len(set(chunk["metadata"]["source"] for chunk in chunks))
        diversity_bonus = min(0.15, (sources - 1) * 0.05)
        
        # Factor 3: Answer characteristics (15% weight)
        answer_length = len(answer.split())
        has_citations = "source" in answer.lower() or "page" in answer.lower()
        answer_confidence = 0.1
        if 50 <= answer_length <= 1000:
            answer_confidence += 0.03
        if has_citations:
            answer_confidence += 0.02
        
        total_confidence = min(1.0, score_confidence + diversity_bonus + answer_confidence)
        self.metrics.avg_confidence = total_confidence
        
        return total_confidence
    
    def generate_grounded_answer(
        self, 
        question: str, 
        context_prompt: str, 
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Generate complete grounded answer with citations and confidence
        
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
            confidence = self.calculate_confidence(chunks, answer)
            
            return answer, citations, confidence
            
        except Exception as e:
            logger.error(f"Error generating grounded answer: {str(e)}")
            raise
    
    def format_answer_with_citations(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        format_type: ResponseFormat = ResponseFormat.WITH_CITATIONS
    ) -> Dict[str, Any]:
        """
        Format answer with citations in requested format
        
        Args:
            answer: Generated answer text
            citations: List of citations
            format_type: Desired output format
            
        Returns:
            Formatted response
        """
        if format_type == ResponseFormat.STANDARD:
            return {
                "answer": answer,
                "citations": [f"{c['source']} (Page {c['page']})" for c in citations]
            }
        
        elif format_type == ResponseFormat.WITH_CITATIONS:
            # Embed citations into answer
            formatted = answer
            for idx, citation in enumerate(citations, 1):
                citation_marker = f" [Source {idx}: {citation['source']}, Page {citation['page']}]"
                formatted += citation_marker if idx == 1 else ""
            
            return {
                "answer": formatted,
                "citations": citations,
                "citation_count": len(citations)
            }
        
        elif format_type == ResponseFormat.JSON:
            return {
                "answer": answer,
                "citations": citations,
                "format": "json"
            }
        
        elif format_type == ResponseFormat.BULLET_POINTS:
            # Split answer into sentences
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            return {
                "answer_points": sentences,
                "citations": citations,
                "point_count": len(sentences)
            }
        
        return {"answer": answer, "citations": citations}
    
    def generate_comprehensive_response(
        self,
        question: str,
        context_prompt: str,
        chunks: List[Dict[str, Any]],
        response_format: ResponseFormat = ResponseFormat.WITH_CITATIONS,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive response with all metadata
        
        Args:
            question: User's question
            context_prompt: Formatted prompt
            chunks: Retrieved chunks
            response_format: Output format
            include_metadata: Include retrieval metadata
            
        Returns:
            Complete response with answer, citations, confidence, metadata
        """
        try:
            # Generate grounded answer
            answer, citations, confidence = self.generate_grounded_answer(
                question, 
                context_prompt, 
                chunks
            )
            
            # Format answer
            formatted = self.format_answer_with_citations(answer, citations, response_format)
            
            # Build complete response
            response = {
                "question": question,
                "answer": formatted.get("answer", answer),
                "citations": citations,
                "confidence": confidence,
                "response_format": response_format.value,
                "generated_at": datetime.utcnow().isoformat(),
            }
            
            if include_metadata:
                response["metadata"] = {
                    "chunks_used": len(chunks),
                    "unique_sources": len(set(c["source"] for c in citations)),
                    "model": self.model_name,
                    "grounding_level": self.grounding_level,
                    "retrieval_scores": {
                        "avg": sum(c["similarity_score"] for c in chunks) / len(chunks) if chunks else 0,
                        "min": min(c["similarity_score"] for c in chunks) if chunks else 0,
                        "max": max(c["similarity_score"] for c in chunks) if chunks else 0,
                    }
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            raise
    
    def validate_grounding(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that answer is properly grounded in context
        
        Args:
            answer: Generated answer
            chunks: Retrieved context chunks
            
        Returns:
            Validation report
        """
        # Check for common ungrounded phrases
        ungrounded_phrases = [
            "I believe", "I think", "it is generally known",
            "as everyone knows", "obviously", "of course"
        ]
        
        found_ungrounded = [phrase for phrase in ungrounded_phrases if phrase.lower() in answer.lower()]
        
        # Check for citations
        has_citations = any(
            f"{c['source']}" in answer or f"Page {c['page']}" in answer 
            for c in chunks
        )
        
        # Calculate grounding score
        grounding_score = 1.0
        if found_ungrounded:
            grounding_score -= 0.2 * len(found_ungrounded)
        if not has_citations and chunks:
            grounding_score -= 0.3
        
        grounding_score = max(0.0, min(1.0, grounding_score))
        
        return {
            "is_properly_grounded": grounding_score >= 0.7,
            "grounding_score": grounding_score,
            "ungrounded_phrases_found": found_ungrounded,
            "has_citations": has_citations,
            "warning_messages": [
                f"Found ungrounded phrase: '{phrase}'" for phrase in found_ungrounded
            ] + (["No source citations found"] if not has_citations and chunks else [])
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated answer generation metrics"""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset metrics for fresh tracking"""
        self.metrics = AnswerGenerationMetrics()

