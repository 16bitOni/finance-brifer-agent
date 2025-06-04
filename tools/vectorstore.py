from pinecone import Pinecone
import os
from dotenv import load_dotenv
import yaml
import time
from typing import Optional, Dict

# Load environment variables
load_dotenv()

def get_vector_store(index_name: str = None) -> Pinecone:
    """
    Initialize and return a Pinecone client with an index using integrated embedding.
    
    Args:
        index_name: Name of the Pinecone index (optional)
    
    Returns:
        Pinecone index instance
    """
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Get API key and environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = config["vector_store"]["environment"]
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key")
    if not pinecone_env:
        raise ValueError("Missing Pinecone environment in config")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Use provided index name or default from config
    index_name = index_name or config["vector_store"]["index_name"]
    
    # Create index with integrated embedding if it doesn't exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region=pinecone_env,
            embed={
                "model": "multilingual-e5-large",
                "field_map": {"text": "chunk_text"}
            }
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    # Get the index
    return pc.Index(index_name)

def upsert_text(index: Pinecone, texts: list, metadata: list = None, namespace: str = ""):
    """
    Upsert text data into Pinecone index using integrated embedding.
    
    Args:
        index: Pinecone index instance
        texts: List of text strings to embed and store
        metadata: Optional list of metadata dictionaries for each text
        namespace: Namespace to upsert into (default: empty string)
    """
    # Prepare records for upsert
    records = []
    for i, text in enumerate(texts):
        record = {
            "_id": f"doc_{i}",
            "chunk_text": text
        }
        if metadata and i < len(metadata):
            # Add metadata fields
            for key, value in metadata[i].items():
                if key != "text":  # Skip text as it's already in chunk_text
                    record[key] = value
        records.append(record)
    
    # Upsert records
    index.upsert_records(namespace, records)
    
    # Wait for vectors to be indexed
    time.sleep(10)

def query_text(index, query_text: str, top_k: int = 5, namespace: str = "") -> Optional[Dict]:
    """
    Query the vector store with text input.
    
    Args:
        index: Pinecone index
        query_text: Text to query
        top_k: Number of results to return
        namespace: Namespace to query (default: "")
        
    Returns:
        Query results or None if query fails
    """
    try:
        results = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query_text},
                "top_k": top_k
            },
            fields=["chunk_text", "source", "company", "date", "type"]
        )
        return results
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return None

def batch_upsert_text(index: Pinecone, texts: list, metadata: list = None, 
                     batch_size: int = 200, namespace: str = ""):
    """
    Upsert text data in batches for better performance.
    
    Args:
        index: Pinecone index instance
        texts: List of text strings to embed and store
        metadata: Optional list of metadata dictionaries for each text
        batch_size: Number of records per batch (default: 200)
        namespace: Namespace to upsert into (default: empty string)
    """
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    # Prepare all records
    records = []
    for i, text in enumerate(texts):
        record = {
            "_id": f"doc_{i}",
            "chunk_text": text
        }
        if metadata and i < len(metadata):
            # Add metadata fields
            for key, value in metadata[i].items():
                if key != "text":  # Skip text as it's already in chunk_text
                    record[key] = value
        records.append(record)
    
    # Upsert in batches
    for batch in chunks(records, batch_size):
        index.upsert_records(namespace, batch)
        time.sleep(1)  # Small delay between batches 