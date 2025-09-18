import os
from config import config
import json
from datetime import datetime

class FileMetadataManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.metadata_file = os.path.join(db_path, config.METADATA_FILE)
        self.ensure_db_directory()
    
    def ensure_db_directory(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
    
    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_file_processed(self, filename, file_hash):
        metadata = self.load_metadata()
        if filename in metadata:
            return metadata[filename].get('hash') == file_hash
        return False
    
    def add_file_metadata(self, filename, file_hash, chunk_count, extra_info = None):
        metadata = self.load_metadata()
        file_info = {
            'hash': file_hash,
            'chunk_count': chunk_count,
            'processed_at': datetime.now().isoformat()
        }
        
        if extra_info:
          file_info.update(extra_info)
        
        metadata[filename] = file_info
        self.save_metadata(metadata)

  
    def get_processed_files(self):
        metadata = self.load_metadata()
        return list(metadata.keys())
