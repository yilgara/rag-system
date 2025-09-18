from sentence_transformers import SentenceTransformer
import os
import pickle
import streamlit as st
import faiss
import numpy as np


class EmbeddingManager:
    def __init__(self, model_name = "all-MiniLM-L6-v2", db_path = "rag_database"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.chunks_file = os.path.join(db_path, "chunks.pkl")
        self.embeddings_file = os.path.join(db_path, "embeddings.npy")
        
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        # Ensure directory exists
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        # Load existing data
        self.load_existing_data()


  
    def load_existing_data(self):
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                    
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)
                
            if self.index and self.chunks:
                st.success(f"Loaded existing database with {len(self.chunks)} chunks")
        except Exception as e:
            st.warning(f"Could not load existing data: {str(e)}")
            self.index = None
            self.chunks = []
            self.embeddings = None


  
    def create_embeddings(self, chunks):
        with st.spinner(f"Creating embeddings for {len(chunks)} new chunks..."):
            embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings


  
    def add_chunks_to_db(self, new_chunks):
        if not new_chunks:
            return
        
        # Create embeddings for new chunks
        new_embeddings = self.create_embeddings(new_chunks)
        
        # Initialize or update index
        if self.index is None:
            # Create new index
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.chunks = []
            self.embeddings = new_embeddings
        else:
            # Add to existing
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(new_embeddings)
        
        # Add to index
        self.index.add(new_embeddings)
        
        # Add chunks
        self.chunks.extend(new_chunks)
        
        # Save to disk
        self.save_to_disk()


  
    def save_to_disk(self):
        try:
            faiss.write_index(self.index, self.index_file)
            
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
                
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
                
        except Exception as e:
            st.error(f"Error saving to disk: {str(e)}")


  
    def search_similar_chunks(self, query, k = 3):
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

  
    def get_database_stats(self):
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }
