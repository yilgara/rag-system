from dataclasses import dataclass

@dataclass
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TOP_K_CHUNKS = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DB_PATH = "rag_database"
    METADATA_FILE = "file_metadata.json"

config = Config()
