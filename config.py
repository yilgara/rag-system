from dataclasses import dataclass

@dataclass
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_CHUNKS = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLAMA_MODEL = "microsoft/DialoGPT-medium"  # Lighter alternative, can be changed to full LLaMA
    DB_PATH = "rag_database"
    METADATA_FILE = "file_metadata.json"

config = Config()
