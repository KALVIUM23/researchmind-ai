import os
import uuid
from pathlib import Path
from datetime import datetime


def generate_document_id() -> str:
    """Generate unique document ID"""
    return str(uuid.uuid4())


def get_upload_path(filename: str) -> Path:
    """Get path for uploaded file"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir / filename


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove dangerous characters
    filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
    # Add timestamp to ensure uniqueness
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}{ext}"


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    return text
