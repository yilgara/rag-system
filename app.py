import streamlit as st
import PyPDF2
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from typing import List, Tuple
import re
import requests
import json
import os
from dataclasses import dataclass

# Configuration
@dataclass
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_CHUNKS = 3
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

config = Config()

class DocumentProcessor:
    """Handles document reading and processing"""
    
    @staticmethod
    def read_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def read_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""

class LlamaChunker:
    """Handles text chunking using Llama-style approach"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using sliding window approach similar to Llama
        """
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(start + self.chunk_size - 200, start)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end != -1:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - self.chunk_overlap
            
            # Avoid infinite loop
            if start <= end - self.chunk_size:
                start = end - self.chunk_overlap + 1
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)\[\]"\']', ' ', text)
        return text.strip()
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the range"""
        sentence_endings = ['.', '!', '?', '\n']
        best_pos = -1
        
        for i in range(end - 1, start - 1, -1):
            if text[i] in sentence_endings:
                # Check if it's not an abbreviation (simple check)
                if text[i] == '.' and i > 0 and text[i-1].isupper() and i < len(text) - 1 and text[i+1].isupper():
                    continue
                best_pos = i + 1
                break
        
        return best_pos

class EmbeddingManager:
    """Manages embeddings and vector database operations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        with st.spinner("Creating embeddings..."):
            embeddings = self.model.encode(chunks, show_progress_bar=False)
        return embeddings
    
    def build_vector_db(self, chunks: List[str]):
        """Build FAISS vector database"""
        self.chunks = chunks
        self.embeddings = self.create_embeddings(chunks)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for most similar chunks"""
        if self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

class LLMManager:
    """Manages LLM interactions for answer generation using Google Gemini"""
    
    def __init__(self):
        self.model = None
    
    def initialize_gemini(self, api_key: str):
        """Initialize Gemini client"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            return False
    
    def generate_answer(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """Generate answer using Gemini model"""
        if not self.model:
            return "Please configure your Gemini API key first."
        
        # Prepare context
        context = "\n\n".join([f"Chunk {i+1} (Relevance: {score:.3f}):\n{chunk}" 
                              for i, (chunk, score) in enumerate(context_chunks)])
        
        prompt = f"""Based on the following context chunks, please answer the user's question. 
If the answer cannot be found in the provided context, please say so clearly.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.1,
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class RAGSystem:
    """Main RAG System class"""
    
    def __init__(self):
        self.chunker = LlamaChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL)
        self.llm_manager = LLMManager()
        self.processed_chunks = []
        self.is_ready = False
    
    def process_documents(self, files) -> bool:
        """Process uploaded documents"""
        all_text = ""
        
        for file in files:
            if file.type == "application/pdf":
                text = DocumentProcessor.read_pdf(file)
            elif file.type == "text/plain":
                text = DocumentProcessor.read_txt(file)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                continue
            
            all_text += f"\n\n--- Document: {file.name} ---\n\n" + text
        
        if not all_text.strip():
            return False
        
        # Chunk the text
        with st.spinner("Chunking documents..."):
            self.processed_chunks = self.chunker.chunk_text(all_text)
        
        # Build vector database
        with st.spinner("Building vector database..."):
            self.embedding_manager.build_vector_db(self.processed_chunks)
        
        self.is_ready = True
        return True
    
    def query(self, question: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Process query and return answer with relevant chunks"""
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

def main():
    st.set_page_config(
        page_title="RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG System - Document Q&A")
    st.markdown("Upload documents and ask questions to get AI-powered answers!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Status
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ Gemini API Key loaded from secrets")
        except KeyError:
            st.error("❌ Gemini API Key not found in secrets")
            st.info("Add GEMINI_API_KEY to your Streamlit secrets")
        
        st.markdown("---")
        
        # System information
        st.header("System Info")
        st.info(f"LLM Model: Google Gemini Pro")
        st.info(f"Chunk Size: {config.CHUNK_SIZE}")
        st.info(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
        st.info(f"Top K Chunks: {config.TOP_K_CHUNKS}")
        st.info(f"Embedding Model: {config.EMBEDDING_MODEL}")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("Process Documents", type="primary"):
                success = st.session_state.rag_system.process_documents(uploaded_files)
                if success:
                    st.success(f"Successfully processed {len(st.session_state.rag_system.processed_chunks)} chunks!")
                else:
                    st.error("Failed to process documents.")
    
    with col2:
        st.header("❓ Ask Questions")
        
        if not st.session_state.rag_system.is_ready:
            st.warning("Please upload and process documents first.")
        else:
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?"
            )
            
            if st.button("Get Answer", type="primary") and question:
                try:
                    # Check if API key is available
                    api_key = st.secrets["GEMINI_API_KEY"]
                    
                    with st.spinner("Generating answer..."):
                        answer, relevant_chunks = st.session_state.rag_system.query(question)
                    
                    # Display results
                    st.header("🤖 Answer")
                    st.write(answer)
                    
                    st.header("📋 Relevant Chunks")
                    for i, (chunk, score) in enumerate(relevant_chunks):
                        with st.expander(f"Chunk {i+1} (Relevance: {score:.3f})"):
                            st.write(chunk)
                
                except KeyError:
                    st.error("Gemini API key not found in secrets. Please add GEMINI_API_KEY to your Streamlit secrets.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit • Uses Sentence Transformers, FAISS, and OpenAI GPT"
    )

if __name__ == "__main__":
    main()

