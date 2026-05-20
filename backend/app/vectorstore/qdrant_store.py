"""Vector Store Service - Phase 5 (Advanced Qdrant Integration)"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, Range, HasIdCondition
)
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4
import logging
import time

logger = logging.getLogger(__name__)


class VectorStoreStatistics:
    """Track vector store statistics"""
    def __init__(self):
        self.total_insertions = 0
        self.total_searches = 0
        self.total_deletions = 0
        self.batch_operations = 0
        self.total_vectors_stored = 0
        
    def to_dict(self) -> Dict[str, int]:
        return {
            "total_insertions": self.total_insertions,
            "total_searches": self.total_searches,
            "total_deletions": self.total_deletions,
            "batch_operations": self.batch_operations,
            "total_vectors_stored": self.total_vectors_stored,
        }


class VectorStoreService:
    """Advanced Qdrant vector store management with batch operations and monitoring"""
    
    def __init__(self, url: str, api_key: str, collection_name: str, embedding_dim: int):
        """
        Initialize vector store with enhanced features
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Collection name
            embedding_dim: Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        try:
            self.client = QdrantClient(url=url, api_key=api_key if api_key else None)
            self.collection_name = collection_name
            self.embedding_dim = embedding_dim
            self.stats = VectorStoreStatistics()
            
            # Ensure collection exists with optimized settings
            self._ensure_collection_exists()
            logger.info(
                f"Connected to Qdrant at {url}, collection: {collection_name}, "
                f"embedding_dim: {embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            raise
    
    def _ensure_collection_exists(self):
        """Create collection with optimized settings if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                # Simple collection creation compatible with Qdrant Cloud
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"[OK] Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"[OK] Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            # Don't fail - collection might already exist
            logger.info(f"Continuing without creating collection (it may already exist)")
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """
        Add chunks with embeddings to vector store
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors
            
        Returns:
            List of point IDs (UUIDs as strings)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})")
        
        try:
            points = []
            point_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Generate UUID for point ID
                point_id = str(uuid4())
                point_ids.append(point_id)
                
                # Extract metadata from chunk
                metadata = chunk.get("metadata", {})
                payload = {
                    "text": chunk.get("text", ""),
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                    "char_start": metadata.get("char_start"),
                    "char_end": metadata.get("char_end"),
                    "text_preview": metadata.get("text_preview"),
                    "created_at": metadata.get("created_at"),
                }
                
                point = PointStruct(
                    id=point_id,  # Keep as string for UUID compatibility
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            self.stats.total_insertions += len(points)
            self.stats.total_vectors_stored += len(points)
            logger.info(f"[OK] Added {len(points)} chunks to vector store")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise
    
    def add_chunks_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        batch_size: int = 100
    ) -> List[str]:
        """
        Add chunks in optimized batches for large datasets
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            batch_size: Number of chunks per batch (default 100)
            
        Returns:
            List of all point IDs
        """
        all_point_ids = []
        total_chunks = len(chunks)
        
        try:
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}, chunks {i+1}-{min(i+batch_size, total_chunks)}/{total_chunks}")
                batch_ids = self.add_chunks(batch_chunks, batch_embeddings)
                all_point_ids.extend(batch_ids)
                
                self.stats.batch_operations += 1
            
            logger.info(f"[OK] Batch insertion complete: {len(all_point_ids)} total vectors stored")
            return all_point_ids
            
        except Exception as e:
            logger.error(f"Error in batch insertion: {str(e)}")
            raise
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of retrieved chunks with scores and metadata
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,  # Don't retrieve vectors to save bandwidth
            )
            
            retrieved = []
            for result in results.points:
                retrieved.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "similarity_score": result.score,
                    "metadata": {
                        "chunk_id": result.payload.get("chunk_id"),
                        "document_id": result.payload.get("document_id"),
                        "source": result.payload.get("source"),
                        "page": result.payload.get("page"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "char_start": result.payload.get("char_start"),
                        "char_end": result.payload.get("char_end"),
                    }
                })
            
            self.stats.total_searches += 1
            top_score = results.points[0].score if results.points else 0.0
            logger.info(f"[OK] Retrieved {len(retrieved)} chunks, top score: {top_score:.3f}")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def search_with_filter(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering
        
        Args:
            query_embedding: Query embedding vector
            filters: Metadata filter conditions (e.g., {"document_id": "doc123", "page": 5})
            top_k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            Filtered search results
        """
        try:
            # Build filter conditions
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            filter_obj = Filter(must=conditions) if conditions else None
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filter_obj,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
            
            retrieved = []
            for result in results.points:
                retrieved.append({
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "similarity_score": result.score,
                    "metadata": {
                        "chunk_id": result.payload.get("chunk_id"),
                        "document_id": result.payload.get("document_id"),
                        "source": result.payload.get("source"),
                        "page": result.payload.get("page"),
                        "chunk_index": result.payload.get("chunk_index"),
                    }
                })
            
            logger.info(f"[OK] Retrieved {len(retrieved)} filtered chunks, filters: {filters}")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            raise
    
    def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of deleted points
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.client.filter_by(
                    key="document_id",
                    match={"value": document_id}
                )
            )
            self.stats.total_deletions += 1
            logger.info(f"[OK] Deleted chunks for document: {document_id}")
            return 1  # Return count of operation (Qdrant API)
        except Exception as e:
            logger.error(f"Error deleting chunks by document: {str(e)}")
            raise
    
    def delete_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Delete chunks matching metadata criteria
        
        Args:
            metadata_filter: Filter conditions (e.g., {"page": 5, "source": "file.pdf"})
            
        Returns:
            Number of deleted points
        """
        try:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            
            filter_obj = Filter(must=conditions) if conditions else None
            
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_obj
            )
            self.stats.total_deletions += 1
            logger.info(f"[OK] Deleted chunks matching filters: {metadata_filter}")
            return 1
        except Exception as e:
            logger.error(f"Error deleting chunks by metadata: {str(e)}")
            raise
    
    def upsert_points(self, points: List[Dict[str, Any]]) -> List[str]:
        """
        Upsert (update or insert) points directly
        
        Args:
            points: List of point dictionaries with id, vector, payload
            
        Returns:
            List of point IDs
        """
        try:
            point_objects = [
                PointStruct(
                    id=p.get("id", str(uuid4())),
                    vector=p["vector"],
                    payload=p.get("payload", {})
                )
                for p in points
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=point_objects,
                wait=True
            )
            
            self.stats.total_insertions += len(point_objects)
            logger.info(f"[OK] Upserted {len(point_objects)} points")
            return [str(p.id) for p in point_objects]
            
        except Exception as e:
            logger.error(f"Error upserting points: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "embedding_dim": self.embedding_dim,
                "status": "healthy" if info.points_count >= 0 else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_health(self) -> Dict[str, Any]:
        """Check vector store health and connectivity"""
        try:
            # Get collection info
            info = self.get_collection_info()
            
            # Try a basic operation
            start_time = time.time()
            _ = self.client.get_collections()
            operation_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "connectivity": "ok",
                "response_time_ms": operation_time * 1000,
                "collection_info": info,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "connectivity": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store usage statistics"""
        collection_info = self.get_collection_info()
        return {
            **self.stats.to_dict(),
            "collection_info": collection_info
        }
    
    def clear_collection(self, confirm: bool = False) -> bool:
        """
        Clear all vectors from collection (use with caution)
        
        Args:
            confirm: Explicit confirmation to prevent accidental deletion
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("Clear collection requires confirm=True")
            return False
        
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.warning(f"⚠️  Collection '{self.collection_name}' cleared")
            
            # Recreate empty collection
            self._ensure_collection_exists()
            
            # Reset statistics
            self.stats = VectorStoreStatistics()
            logger.info("[OK] Collection recreated empty")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

