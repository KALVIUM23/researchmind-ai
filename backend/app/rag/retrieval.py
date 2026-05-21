"""Retrieval Pipeline - Phase 6 (Advanced Semantic Search & Ranking)"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from .embeddings import EmbeddingsService
from ..vectorstore.qdrant_store import VectorStoreService
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Ranking strategies for retrieved chunks"""
    SIMILARITY_ONLY = "similarity_only"  # Pure similarity score
    DIVERSITY_AWARE = "diversity_aware"  # Balance similarity with source diversity
    PAGE_PROXIMITY = "page_proximity"     # Prefer chunks from same page
    RECENCY = "recency"                   # Prefer newer chunks


class RetrievalMetrics:
    """Track retrieval performance metrics"""
    def __init__(self):
        self.total_queries = 0
        self.total_chunks_retrieved = 0
        self.avg_similarity_score = 0.0
        self.queries_with_filters = 0
        self.deduplication_removals = 0
        self.unique_sources = set()
        self.processing_times = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "avg_similarity_score": self.avg_similarity_score,
            "queries_with_filters": self.queries_with_filters,
            "deduplication_removals": self.deduplication_removals,
            "unique_sources": len(self.unique_sources),
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
        }


class RetrievalService:
    """Handle document retrieval with advanced ranking and filtering"""
    
    def __init__(self, embeddings_service: EmbeddingsService, vector_store: VectorStoreService):
        """
        Initialize retrieval service
        
        Args:
            embeddings_service: Service for generating embeddings
            vector_store: Vector store service
        """
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.metrics = RetrievalMetrics()
    
    def retrieve_context(
        self, 
        question: str, 
        top_k: int = 5,
        document_id: Optional[str] = None,
        min_score: float = 0.0,
        ranking_strategy: RankingStrategy = RankingStrategy.SIMILARITY_ONLY
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks with advanced ranking
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            document_id: Optional document filter
            min_score: Minimum similarity threshold
            ranking_strategy: Strategy for ranking results
            
        Returns:
            Ranked list of relevant chunks
        """
        import time
        start_time = time.time()
        
        try:
            # Generate question embedding
            question_embedding = self.embeddings.embed_text(question)
            
            # Search with optional filtering
            if document_id:
                retrieved_chunks = self.vector_store.search_with_filter(
                    query_embedding=question_embedding,
                    filters={"document_id": document_id},
                    top_k=top_k * 2,  # Get more to filter
                    score_threshold=min_score
                )
                self.metrics.queries_with_filters += 1
            else:
                retrieved_chunks = self.vector_store.search(
                    query_embedding=question_embedding,
                    top_k=top_k * 2,
                    score_threshold=min_score
                )
            
            # Deduplicate chunks
            retrieved_chunks = self._deduplicate_chunks(retrieved_chunks)
            
            # Apply ranking strategy
            ranked_chunks = self._rank_chunks(retrieved_chunks, ranking_strategy, top_k)
            
            # Normalize scores to 0-1 range
            ranked_chunks = self._normalize_scores(ranked_chunks)
            
            # DEBUG: Print retrieved chunks to verify semantic retrieval is working
            print("\n" + "="*80)
            print("[DEBUG] RETRIEVED CHUNKS FROM VECTOR DB")
            print("="*80)
            print(f"Question: {question}")
            print(f"Total chunks retrieved: {len(ranked_chunks)}")
            for i, chunk in enumerate(ranked_chunks[:top_k], 1):
                print(f"\n--- Chunk {i} ---")
                print(f"Similarity Score: {chunk.get('similarity_score', 'N/A'):.4f}")
                print(f"Source: {chunk.get('metadata', {}).get('source', 'Unknown')}")
                print(f"Document ID: {chunk.get('metadata', {}).get('document_id', 'Unknown')}")
                print(f"Text: {chunk.get('text', 'N/A')[:200]}...")
            print("="*80 + "\n")
            logger.info(f"[DEBUG] Retrieved {len(ranked_chunks)} chunks - showing first {min(top_k, len(ranked_chunks))} for question")
            
            # Update metrics
            self.metrics.total_queries += 1
            self.metrics.total_chunks_retrieved += len(ranked_chunks)
            if ranked_chunks:
                scores = [c["similarity_score"] for c in ranked_chunks]
                self.metrics.avg_similarity_score = sum(scores) / len(scores)
            for chunk in ranked_chunks:
                self.metrics.unique_sources.add(chunk["metadata"]["source"])
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics.processing_times.append(processing_time)
            
            logger.info(
                f"[OK] Retrieved {len(ranked_chunks)} chunks for question (strategy: {ranking_strategy.value}, "
                f"time: {processing_time:.1f}ms)"
            )
            return ranked_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on text similarity"""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            # Normalize text for comparison
            text_hash = hash(chunk["text"][:100])  # Use first 100 chars
            
            if text_hash not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(text_hash)
            else:
                self.metrics.deduplication_removals += 1
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks
    
    def _rank_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        strategy: RankingStrategy,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank chunks using specified strategy"""
        if strategy == RankingStrategy.SIMILARITY_ONLY:
            # Sort by similarity score (already sorted from search)
            return sorted(chunks, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
        elif strategy == RankingStrategy.DIVERSITY_AWARE:
            # Balance similarity with source diversity
            ranked = []
            source_count = defaultdict(int)
            
            for chunk in sorted(chunks, key=lambda x: x["similarity_score"], reverse=True):
                source = chunk["metadata"]["source"]
                if source_count[source] < 2:  # Max 2 chunks per source
                    ranked.append(chunk)
                    source_count[source] += 1
                    if len(ranked) >= top_k:
                        break
            
            return ranked
        
        elif strategy == RankingStrategy.PAGE_PROXIMITY:
            # Group by page and prefer chunks from same page
            by_page = defaultdict(list)
            for chunk in chunks:
                page = chunk["metadata"]["page"]
                by_page[page].append(chunk)
            
            ranked = []
            for page in sorted(by_page.keys()):
                ranked.extend(sorted(by_page[page], key=lambda x: x["similarity_score"], reverse=True))
                if len(ranked) >= top_k:
                    break
            
            return ranked[:top_k]
        
        elif strategy == RankingStrategy.RECENCY:
            # Prefer recently created chunks
            return sorted(chunks, key=lambda x: (
                x["metadata"].get("created_at", ""),
                x["similarity_score"]
            ), reverse=True)[:top_k]
        
        else:
            return sorted(chunks, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
    
    def _normalize_scores(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize similarity scores to 0-1 range"""
        if not chunks:
            return []
        
        scores = [c["similarity_score"] for c in chunks]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for chunk in chunks:
                chunk["similarity_score"] = 1.0
        else:
            for chunk in chunks:
                normalized = (chunk["similarity_score"] - min_score) / (max_score - min_score)
                chunk["similarity_score"] = normalized
        
        return chunks
    
    def format_context(
        self, 
        chunks: List[Dict[str, Any]], 
        include_scores: bool = True
    ) -> str:
        """
        Format retrieved chunks into context string for LLM
        
        Args:
            chunks: List of retrieved chunks
            include_scores: Include similarity scores in output
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            score_str = f" | Score: {chunk['similarity_score']:.2f}" if include_scores else ""
            
            source_info = f"[Chunk {idx}: {metadata['source']} | Page {metadata['page']}{score_str}]"
            context_parts.append(f"{source_info}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def prepare_rag_input(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]],
        include_answer_instructions: bool = True
    ) -> str:
        """
        Prepare final RAG input with context and question
        
        Args:
            question: User's question
            chunks: Retrieved context chunks
            include_answer_instructions: Include formatting instructions
            
        Returns:
            Formatted prompt for LLM
        """
        context = self.format_context(chunks, include_scores=False)
        
        instructions = ""
        if include_answer_instructions:
            instructions = """INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the answer is not in context, respond: "The answer is not provided in the available documents"
3. Be concise and factual
4. Always cite the source document and page number
5. Separate answer from citations clearly

"""
        
        prompt = f"""You are a research assistant. Answer the following question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

{instructions}ANSWER:"""
        
        return prompt
    
    def prepare_rag_input_with_formatting(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        response_format: str = "standard"
    ) -> str:
        """
        Prepare RAG input with specific response format
        
        Args:
            question: User's question
            chunks: Retrieved chunks
            response_format: Format type (standard, json, bullet_points)
            
        Returns:
            Formatted prompt
        """
        context = self.format_context(chunks, include_scores=False)
        
        format_instructions = {
            "standard": "Provide a clear, paragraph-based answer.",
            "json": "Provide answer as JSON with 'answer' and 'key_points' fields.",
            "bullet_points": "Provide answer as bullet points with main points highlighted."
        }
        
        instruction = format_instructions.get(response_format, format_instructions["standard"])
        
        prompt = f"""You are a research assistant.

CONTEXT:
{context}

QUESTION: {question}

FORMAT: {instruction}

ANSWER:"""
        
        return prompt
    
    def get_retrieval_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed statistics about retrieval results"""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_score": 0,
                "min_score": 0,
                "max_score": 0,
                "sources": [],
                "pages_covered": 0,
                "documents": []
            }
        
        scores = [chunk["similarity_score"] for chunk in chunks]
        sources = set(chunk["metadata"]["source"] for chunk in chunks)
        pages = set(chunk["metadata"]["page"] for chunk in chunks)
        documents = set(chunk["metadata"]["document_id"] for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "sources": list(sources),
            "pages_covered": len(pages),
            "documents": list(documents),
            "scores_distribution": {
                "high": len([s for s in scores if s >= 0.8]),
                "medium": len([s for s in scores if 0.5 <= s < 0.8]),
                "low": len([s for s in scores if s < 0.5])
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated retrieval metrics"""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset metrics for fresh tracking"""
        self.metrics = RetrievalMetrics()

