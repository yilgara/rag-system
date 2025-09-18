from config import config
from chunker import LlamaChunker
from embedding import EmbeddingManager
from llm import LLMManager
from file_metadata import FileMetadataManager
import streamlit as st
import os
from document_processor import DocumentProcessor


class RAGSystem:
    def __init__(self, db_path = None):
        self.db_path = db_path or config.DB_PATH
        self.chunker = LlamaChunker()
        self.embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL, self.db_path)
        self.llm_manager = LLMManager()
        self.metadata_manager = FileMetadataManager(self.db_path)
        
        # Check if we have existing data
        self.is_ready = len(self.embedding_manager.chunks) > 0

  
    def process_new_documents(self, files):
        results = {
            'new_files': [],
            'skipped_files': [],
            'new_chunks': 0,
            'total_chunks': 0
        }
        
        all_new_chunks = []
        
        for file in files:
            # Read file and get hash
            if file.type == "application/pdf":
                text, file_hash = DocumentProcessor.read_pdf(file)
            elif file.type == "text/plain":
                text, file_hash = DocumentProcessor.read_txt(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue
            
            # Check if file already processed
            if self.metadata_manager.is_file_processed(file.name, file_hash):
                results['skipped_files'].append(file.name)
                continue
            
            # Process new file
            if text.strip():
                with st.spinner(f"Processing {file.name} with chunking..."):
                    file_chunks = self.chunker.chunk_text_with_llama(text)
                    
                    # Get detailed metadata for display
                    chunk_metadata = self.chunker.get_chunk_metadata(text)
                
                # Add document identifier to chunks
                prefixed_chunks = [f"[Document: {file.name}]\n{chunk}" for chunk in file_chunks]
                all_new_chunks.extend(prefixed_chunks)
                
                # Update metadata with chunking info
                self.metadata_manager.add_file_metadata(
                    file.name, 
                    file_hash, 
                    len(file_chunks),
                    extra_info={
                        'chunking_method': 'spacy_intelligent',
                        'avg_sentences_per_chunk': sum(c['sentence_count'] for c in chunk_metadata) / len(chunk_metadata) if chunk_metadata else 0,
                        'total_sentences': sum(c['sentence_count'] for c in chunk_metadata),
                        'paragraphs_processed': len(set(c['paragraph_index'] for c in chunk_metadata))
                    }
                )
                results['new_files'].append(file.name)
                results['new_chunks'] += len(file_chunks)
        
        # Add all new chunks to database
        if all_new_chunks:
            self.embedding_manager.add_chunks_to_db(all_new_chunks)
            self.is_ready = True
        
        # Update total chunks count
        results['total_chunks'] = len(self.embedding_manager.chunks)
        
        return results
    
    def query(self, question):
        if not self.is_ready:
            return "Please upload and process documents first.", []
        
        # Get Gemini API key from secrets
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            return "Gemini API key not found in secrets. Please add GEMINI_API_KEY to your Streamlit secrets.", []
        
        # Initialize LLM
        if not self.llm_manager.initialize_gemini(api_key):
            return "Failed to initialize Gemini API. Please check your API key.", []
        
        # Find relevant chunks
        relevant_chunks = self.embedding_manager.search_similar_chunks(
            question, config.TOP_K_CHUNKS
        )
        
        # Generate answer
        answer = self.llm_manager.generate_answer(question, relevant_chunks)
        
        return answer, relevant_chunks

  
    def get_system_stats(self):
        stats = self.embedding_manager.get_database_stats()
        stats['processed_files'] = self.metadata_manager.get_processed_files()
        return stats

  
    def reset_database(self):
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.__init__()  # Reinitialize
