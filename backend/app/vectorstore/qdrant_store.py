"""Vector Store Service - Stage 4 (Qdrant Integration)"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manage vector storage in Qdrant"""
    
    def __init__(self, url: str, api_key: str, collection_name: str, embedding_dim: int):
        """
        Initialize vector store
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key
            collection_name: Collection name
            embedding_dim: Embedding dimension
        """
        try:
            self.client = QdrantClient(url=url, api_key=api_key if api_key else None)
            self.collection_name = collection_name
            self.embedding_dim = embedding_dim
            
            # Ensure collection exists
            self._ensure_collection_exists()
            logger.info(f"Connected to Qdrant at {url}, collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            raise
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    ),
                )
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """
        Add chunks with embeddings to vector store
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors
            
        Returns:
            List of point IDs
        """
        try:
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=hash(chunk["text"]) % (2**31),  # Simple hash for ID
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
                points.append(point)
            
            # Upsert points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Added {len(points)} chunks to vector store")
            return [str(p.id) for p in points]
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            
            retrieved = []
            for result in results:
                retrieved.append({
                    "text": result.payload["text"],
                    "metadata": {
                        "source": result.payload.get("source"),
                        "page": result.payload.get("page"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "document_id": result.payload.get("document_id"),
                    },
                    "similarity_score": result.score
                })
            
            logger.info(f"Retrieved {len(retrieved)} chunks, top score: {results[0].score if results else 0:.3f}")
            return retrieved
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def delete_by_document(self, document_id: str):
        """Delete all chunks for a document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.client.filter_by(
                    key="document_id",
                    match={"value": document_id}
                )
            )
            logger.info(f"Deleted chunks for document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
