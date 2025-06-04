import os
import json
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import streamlit as st

from tools.vectorstore import get_vector_store, batch_upsert_text

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class EmbeddingService:
    """Service for handling document embedding and storage."""
    
    def __init__(self):
        self.index = get_vector_store()
        self.chunk_size = config["embedding"]["chunk_size"]
        self.chunk_overlap = config["embedding"]["chunk_overlap"]
        self.batch_size = config["embedding"]["batch_size"]
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def prepare_metadata(self, source: str, content_type: str, additional_metadata: Optional[Dict] = None) -> Dict:
        """Prepare metadata for embedding."""
        metadata = {
            "source": source,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
        }
        if additional_metadata:
            metadata.update(additional_metadata)
        return metadata
    
    def embed_text(self, text: str, metadata: Dict[str, Any], namespace: str = "") -> None:
        """Embed a single text with metadata."""
        chunks = self.chunk_text(text)
        metadatas = [metadata] * len(chunks)
        batch_upsert_text(self.index, chunks, metadatas, namespace=namespace)
    
    def embed_file(self, file_path: str, content_type: str, namespace: str = "") -> None:
        """Embed a file's contents."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file based on extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                text = row.to_string()
                metadata = self.prepare_metadata(
                    str(file_path),
                    content_type,
                    {"row_index": _}
                )
                self.embed_text(text, metadata, namespace)
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        text = json.dumps(item)
                        metadata = self.prepare_metadata(
                            str(file_path),
                            content_type,
                            {"item_index": i}
                        )
                        self.embed_text(text, metadata, namespace)
                else:
                    text = json.dumps(data)
                    metadata = self.prepare_metadata(str(file_path), content_type)
                    self.embed_text(text, metadata, namespace)
        
        elif file_path.suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                metadata = self.prepare_metadata(str(file_path), content_type)
                self.embed_text(text, metadata, namespace)
        
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def embed_mock_data(self, namespace: str = "") -> None:
        """Embed mock portfolio data."""
        mock_data = {
    "portfolio": {
        "holdings": [
            {"name": "Apple", "symbol": "AAPL", "shares": 100, "avg_price": 150.0, "sector": "Technology", "region": "US"},
            {"name": "Microsoft", "symbol": "MSFT", "shares": 50, "avg_price": 280.0, "sector": "Technology", "region": "US"},
            {"name": "Taiwan Semiconductor", "symbol": "TSM", "shares": 200, "avg_price": 80.0, "sector": "Technology", "region": "Asia"},
            {"name": "Samsung Electronics", "symbol": "005930.KS", "shares": 150, "avg_price": 1200.0, "sector": "Technology", "region": "Asia"},
            {"name": "JPMorgan Chase", "symbol": "JPM", "shares": 75, "avg_price": 120.0, "sector": "Financial", "region": "US"},
            {"name": "Johnson & Johnson", "symbol": "JNJ", "shares": 200, "avg_price": 160.0, "sector": "Healthcare", "region": "US"},
            {"name": "Alibaba Group", "symbol": "9988.HK", "shares": 80, "avg_price": 85.0, "sector": "Technology", "region": "Asia"},
            {"name": "PDD Holdings", "symbol": "PDD", "shares": 120, "avg_price": 95.0, "sector": "Technology", "region": "Asia"}
        ],
        "cash": 50000.0,
        "last_updated": datetime.now().isoformat()
    }
}
        
        
        # Convert mock data to text and embed
        text = json.dumps(mock_data, indent=2)
        metadata = self.prepare_metadata("mock_data", "portfolio")
        self.embed_text(text, metadata, namespace)
    
    def embed_streamlit_upload(self, uploaded_file, content_type: str, namespace: str = "") -> None:
        """Handle file uploads from Streamlit."""
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                self.embed_file(tmp_path, content_type, namespace)
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

def initialize_mock_data():
    """Initialize the system with mock data."""
    service = EmbeddingService()
    service.embed_mock_data()
    return "Mock data embedded successfully!"

def embed_user_data(file_path: str, content_type: str, namespace: str = ""):
    """Embed user-provided data."""
    service = EmbeddingService()
    service.embed_file(file_path, content_type, namespace)
    return "User data embedded successfully!"

def embed_streamlit_data(uploaded_file, content_type: str, namespace: str = ""):
    """Embed data uploaded through Streamlit."""
    service = EmbeddingService()
    service.embed_streamlit_upload(uploaded_file, content_type, namespace)
    return "Uploaded data embedded successfully!" 