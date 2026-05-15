import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def log_upload(filename: str, pages: int):
    """Log file upload"""
    logger.info(f"PDF uploaded: {filename} ({pages} pages) at {datetime.now()}")


def log_retrieval(question: str, chunks_retrieved: int, similarity_scores: list):
    """Log retrieval operation"""
    avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    logger.info(f"Retrieved {chunks_retrieved} chunks for: '{question}' (avg similarity: {avg_score:.3f})")


def log_answer_generation(question: str, answer_length: int):
    """Log answer generation"""
    logger.info(f"Generated answer ({answer_length} chars) for: '{question}'")
