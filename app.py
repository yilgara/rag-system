import streamlit as st
import os
import hashlib
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Document processing
import PyPDF2
import docx
import csv
from io import StringIO

# Vector database and embeddings
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Text processing
import nltk
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Safely try to import ChromaDB
def try_import_chromadb():
    """Safely import ChromaDB, return None if not available"""
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb, Settings
    except (ImportError, RuntimeError, Exception) as e:
        return None, None

# Check ChromaDB availability
chromadb, Settings = try_import_chromadb()
CHROMADB_AVAILABLE = chromadb is not None

class DocumentProcessor:
    """Handles document parsing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunks.append({
                            'text': text,
                            'page': page_num + 1,
                            'type': 'page'
                        })
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        return chunks
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
        """Extract text from DOCX with paragraph information"""
        chunks = []
        try:
            doc = docx.Document(file_path)
            for para_num, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if text:
                    chunks.append({
                        'text': text,
                        'paragraph': para_num + 1,
                        'type': 'paragraph'
                    })
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
        return chunks
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
        """Extract text from TXT with line information"""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line_num, line in enumerate(lines):
                    text = line.strip()
                    if text:
                        chunks.append({
                            'text': text,
                            'line': line_num + 1,
                            'type': 'line'
                        })
        except Exception as e:
            st.error(f"Error processing TXT: {str(e)}")
        return chunks
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> List[Dict[str, Any]]:
        """Extract text from CSV with row information"""
        chunks = []
        try:
            df = pd.read_csv(file_path)
            for row_num, row in df.iterrows():
                # Convert row to text
                text = ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                if text:
                    chunks.append({
                        'text': text,
                        'row': row_num + 1,
                        'type': 'row'
                    })
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
        return chunks

class TextChunker:
    """Handles text chunking with overlap"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_length = len(sent.split())
                    if overlap_length + sent_length <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_length
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class VectorDatabase:
    """Abstract vector database interface"""
    
    def __init__(self, db_path: str, embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.metadata_file = os.path.join(db_path, "vector_metadata.json")
        os.makedirs(db_path, exist_ok=True)
        
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict], ids: List[str]):
        raise NotImplementedError
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        raise NotImplementedError
    
    def delete(self, ids: List[str]):
        raise NotImplementedError
    
    def get(self, where: Dict = None, include: List[str] = None):
        raise NotImplementedError

class FAISSDatabase(VectorDatabase):
    """FAISS-based vector database implementation"""
    
    def __init__(self, db_path: str, embedding_dim: int = 384):
        super().__init__(db_path, embedding_dim)
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.documents_file = os.path.join(db_path, "documents.pkl")
        self.metadatas_file = os.path.join(db_path, "metadatas.pkl")
        self.ids_file = os.path.join(db_path, "ids.pkl")
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Load existing data
        self.load_data()
    
    def load_data(self):
        """Load existing index and metadata"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            
            if os.path.exists(self.documents_file):
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                self.documents = []
            
            if os.path.exists(self.metadatas_file):
                with open(self.metadatas_file, 'rb') as f:
                    self.metadatas = pickle.load(f)
            else:
                self.metadatas = []
            
            if os.path.exists(self.ids_file):
                with open(self.ids_file, 'rb') as f:
                    self.ids = pickle.load(f)
            else:
                self.ids = []
                
        except Exception as e:
            print(f"Error loading FAISS data: {e}")
            self.documents = []
            self.metadatas = []
            self.ids = []
    
    def save_data(self):
        """Save index and metadata"""
        try:
            faiss.write_index(self.index, self.index_file)
            
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(self.metadatas_file, 'wb') as f:
                pickle.dump(self.metadatas, f)
            
            with open(self.ids_file, 'wb') as f:
                pickle.dump(self.ids, f)
                
        except Exception as e:
            print(f"Error saving FAISS data: {e}")
    
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict], ids: List[str]):
        """Add vectors to the index"""
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Save data
        self.save_data()
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        """Query the index"""
        if self.index.ntotal == 0:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Normalize query embeddings
        query_array = np.array(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))
        
        # Format results
        documents = []
        metadatas = []
        result_distances = []
        
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            docs = []
            metas = []
            dists = []
            
            for dist, idx in zip(dist_row, idx_row):
                if idx != -1:  # Valid index
                    docs.append(self.documents[idx])
                    metas.append(self.metadatas[idx])
                    dists.append(1.0 - dist)  # Convert similarity back to distance
            
            documents.append(docs)
            metadatas.append(metas)
            result_distances.append(dists)
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'distances': result_distances
        }
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs"""
        # Find indices to remove
        indices_to_remove = []
        for id_to_remove in ids:
            if id_to_remove in self.ids:
                idx = self.ids.index(id_to_remove)
                indices_to_remove.append(idx)
        
        if not indices_to_remove:
            return
        
        # Remove from metadata lists (in reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.metadatas[idx]
            del self.ids[idx]
        
        # Rebuild FAISS index (FAISS doesn't support efficient deletion)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.save_data()
        return True
    
    def get(self, where: Dict = None, include: List[str] = None):
        """Get vectors by metadata filter"""
        result_ids = []
        result_metadatas = []
        
        for i, metadata in enumerate(self.metadatas):
            if where is None:
                result_ids.append(self.ids[i])
                result_metadatas.append(metadata)
            else:
                # Simple metadata matching
                match = True
                for key, value in where.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    result_ids.append(self.ids[i])
                    result_metadatas.append(metadata)
        
        return {
            'ids': result_ids,
            'metadatas': result_metadatas
        }

class ChromaDatabase(VectorDatabase):
    """ChromaDB-based vector database implementation"""
    
    def __init__(self, db_path: str, embedding_dim: int = 384):
        super().__init__(db_path, embedding_dim)
        
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB is not available")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict], ids: List[str]):
        """Add vectors to ChromaDB"""
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        """Query ChromaDB"""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
    
    def delete(self, ids: List[str]):
        """Delete from ChromaDB"""
        self.collection.delete(ids=ids)
        return True
    
    def get(self, where: Dict = None, include: List[str] = None):
        """Get from ChromaDB"""
        return self.collection.get(
            where=where,
            include=include or ["metadatas"]
        )

class RAGSystem:
    """Main RAG system class"""
    
    def __init__(self):
        self.db_path = "rag_database"
        self.metadata_file = "file_metadata.json"
        
        # Initialize vector database (try ChromaDB first, fallback to FAISS)
        try:
            if CHROMADB_AVAILABLE:
                self.vector_db = ChromaDatabase(self.db_path)
                st.success("✅ Using ChromaDB for vector storage")
            else:
                raise RuntimeError("ChromaDB not available")
        except Exception as e:
            self.vector_db = FAISSDatabase(self.db_path)
            st.warning("⚠️ Using FAISS for vector storage (ChromaDB not available)")
            if "sqlite3" in str(e).lower():
                st.info("💡 ChromaDB requires SQLite 3.35+. FAISS works great as an alternative!")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Store embedding dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize document processor and chunker
        self.doc_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        
        # Load metadata
        self.load_metadata()
    
    def load_metadata(self):
        """Load file metadata"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save file metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.md5(file_content).hexdigest()
    
    def process_file(self, uploaded_file) -> bool:
        """Process and index a single file"""
        try:
            # Read file content
            file_content = uploaded_file.read()
            file_hash = self.get_file_hash(file_content)
            
            # Check if file already exists
            if file_hash in self.metadata:
                st.warning(f"File {uploaded_file.name} already exists in the database")
                return False
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Extract text based on file type
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                if file_ext == '.pdf':
                    raw_chunks = self.doc_processor.extract_text_from_pdf(tmp_path)
                elif file_ext == '.docx':
                    raw_chunks = self.doc_processor.extract_text_from_docx(tmp_path)
                elif file_ext == '.txt':
                    raw_chunks = self.doc_processor.extract_text_from_txt(tmp_path)
                elif file_ext == '.csv':
                    raw_chunks = self.doc_processor.extract_text_from_csv(tmp_path)
                else:
                    st.error(f"Unsupported file type: {file_ext}")
                    return False
                
                # Process chunks
                processed_chunks = []
                chunk_id = 0

                st.write("hello222")
                
                for raw_chunk in raw_chunks:
                    # Further chunk the text if it's too long
                    text_chunks = self.text_chunker.chunk_text(raw_chunk['text'])
                    st.write("chunk")
                    
                    for text_chunk in text_chunks:
                        st.write("inner chunk")
                        if text_chunk.strip():
                            chunk_metadata = {
                                'filename': uploaded_file.name,
                                'file_hash': file_hash,
                                'chunk_id': chunk_id,
                                'source_info': raw_chunk
                            }
                            st.write("inner chunk if")
                            
                            processed_chunks.append({
                                'text': text_chunk,
                                'metadata': chunk_metadata,
                                'id': f"{file_hash}_{chunk_id}"
                            })
                            chunk_id += 1
                            st.write("inner chunk proced")
                            st.write(chunk_id)
                st.write("hello2")
                if not processed_chunks:
                    st.error("No text content found in the file")
                    return False
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in processed_chunks]
                embeddings = self.embedding_model.encode(texts)
                
                # Store in vector database
                ids = [chunk['id'] for chunk in processed_chunks]
                metadatas = [chunk['metadata'] for chunk in processed_chunks]
                
                self.vector_db.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Update metadata
                self.metadata[file_hash] = {
                    'filename': uploaded_file.name,
                    'chunk_count': len(processed_chunks),
                    'file_size': len(file_content)
                }
                self.save_metadata()
                
                st.success(f"Successfully processed {uploaded_file.name} ({len(processed_chunks)} chunks)")
                return True
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return False
    
    def delete_file(self, file_hash: str) -> bool:
        """Delete a file from the vector database"""
        try:
            if file_hash not in self.metadata:
                st.error("File not found in database")
                return False
            
            # Get all chunk IDs for this file
            results = self.vector_db.get(
                where={"file_hash": file_hash},
                include=["metadatas"]
            )
            
            if results['ids']:
                # Delete chunks from vector database
                self.vector_db.delete(ids=results['ids'])
                
                # Remove from metadata
                filename = self.metadata[file_hash]['filename']
                del self.metadata[file_hash]
                self.save_metadata()
                
                st.success(f"Successfully deleted {filename}")
                return True
            else:
                st.error("No chunks found for this file")
                return False
                
        except Exception as e:
            st.error(f"Error deleting file: {str(e)}")
            return False
    
    def search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant chunks and generate answer"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search vector database
            results = self.vector_db.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            if not results['documents'][0]:
                return {
                    'answer': "No relevant information found in the uploaded documents.",
                    'sources': []
                }
            
            # Prepare context and sources
            context_chunks = []
            sources = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                context_chunks.append(doc)
                
                # Format source information
                source_info = metadata['source_info']
                if source_info['type'] == 'page':
                    location = f"page {source_info['page']}"
                elif source_info['type'] == 'paragraph':
                    location = f"paragraph {source_info['paragraph']}"
                elif source_info['type'] == 'line':
                    location = f"line {source_info['line']}"
                elif source_info['type'] == 'row':
                    location = f"row {source_info['row']}"
                else:
                    location = "unknown location"
                
                sources.append({
                    'filename': metadata['filename'],
                    'location': location,
                    'content_preview': doc[:100] + "..." if len(doc) > 100 else doc,
                    'relevance_score': 1 - distance  # Convert distance to similarity
                })
            
            # Generate answer using the context
            context = "\n\n".join(context_chunks)
            answer = self.generate_answer(query, context, sources)
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return {
                'answer': "An error occurred while searching for information.",
                'sources': []
            }
    
    def generate_answer(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate answer based on context (simple template-based approach)"""
        # This is a simple template-based approach
        # In a production system, you would use an LLM API like OpenAI GPT-4
        
        if not context.strip():
            return "No relevant information found in the uploaded documents."
        
        # Simple keyword matching and context extraction
        query_words = set(query.lower().split())
        try:
            context_sentences = sent_tokenize(context)
        except LookupError:
            # Fallback to simple sentence splitting
            context_sentences = re.split(r'[.!?]+', context)
            context_sentences = [s.strip() for s in context_sentences if s.strip()]
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Create answer with citations
            answer_parts = []
            for i, sentence in enumerate(relevant_sentences[:3]):  # Limit to top 3 sentences
                if i < len(sources):
                    source = sources[i]
                    citation = f"[Source: {source['filename']}, {source['location']}]"
                    answer_parts.append(f"{sentence} {citation}")
            
            return " ".join(answer_parts)
        else:
            return f"Based on the uploaded documents, I found related information but couldn't generate a specific answer to: '{query}'. Please try rephrasing your question."
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded files"""
        files = []
        for file_hash, metadata in self.metadata.items():
            files.append({
                'hash': file_hash,
                'filename': metadata['filename'],
                'chunks': metadata['chunk_count'],
                'size': metadata['file_size']
            })
        return files

def main():
    st.set_page_config(
        page_title="RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Retrieval-Augmented Generation (RAG) System")
    st.markdown("Upload documents and ask questions with AI-powered answers and citations.")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    rag = st.session_state.rag_system
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 File Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, CSV"
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    st.write(f"Processing {file.name}...")
                    rag.process_file(file)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                st.rerun()
        
        st.divider()
        
        # File list and management
        st.subheader("Uploaded Files")
        files = rag.get_uploaded_files()
        
        if files:
            for file in files:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"📄 **{file['filename']}**")
                        st.caption(f"Chunks: {file['chunks']} | Size: {file['size']:,} bytes")
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{file['hash']}", help="Delete file"):
                            rag.delete_file(file['hash'])
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No files uploaded yet")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Query input
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?",
            key="query_input"
        )
        
        col_search, col_clear = st.columns([1, 1])
        
        with col_search:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process query
        if search_clicked and query:
            if not files:
                st.warning("Please upload some documents first!")
            else:
                with st.spinner("Searching and generating answer..."):
                    result = rag.search_and_answer(query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                
                # Clear input
                st.rerun()
        
        # Display chat history
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 Question:** {chat['query']}")
                st.markdown(f"**🤖 Answer:** {chat['answer']}")
                
                if chat['sources']:
                    with st.expander("📚 Sources", expanded=False):
                        for j, source in enumerate(chat['sources']):
                            st.markdown(f"**{j+1}. {source['filename']}** ({source['location']})")
                            st.caption(f"Relevance: {source['relevance_score']:.2f}")
                            st.text(source['content_preview'])
                            st.divider()
                
                st.divider()
    
    with col2:
        st.header("📊 System Stats")
        
        # Database statistics
        total_files = len(files)
        total_chunks = sum(file['chunks'] for file in files) if files else 0
        total_size = sum(file['size'] for file in files) if files else 0
        
        st.metric("Total Files", total_files)
        st.metric("Total Chunks", total_chunks)
        st.metric("Total Size", f"{total_size:,} bytes")
        
        if files:
            st.subheader("📈 File Distribution")
            file_data = pd.DataFrame(files)
            st.bar_chart(file_data.set_index('filename')['chunks'])
        
        st.divider()
        
        # Help section
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Upload Files**: Use the sidebar to upload PDF, DOCX, TXT, or CSV files
        2. **Process**: Click "Process Files" to index them in the vector database
        3. **Ask Questions**: Type your questions in the main area
        4. **Get Answers**: Receive AI-generated answers with source citations
        5. **Manage Files**: Delete files using the 🗑️ button in the sidebar
        """)

if __name__ == "__main__":
    main()
        

