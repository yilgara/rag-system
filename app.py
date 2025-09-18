import streamlit as st
import PyPDF2
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import spacy
from typing import List, Tuple, Dict
import re
import json
import os
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime
import tempfile





class LlamaChunker:
    """Intelligent text chunker using spaCy for sentence/paragraph boundaries"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.LLAMA_MODEL
        self.tokenizer = None
        self.nlp = None
        self.load_models()
    
    @st.cache_resource
    def load_models(_self):
        """Load spaCy and LLaMA models (cached for efficiency)"""
        success = True
        
        # Load spaCy model with better error handling for Streamlit Cloud
        try:
            st.info("🔄 Loading spaCy model...")
            
            # Method 1: Try loading directly (if installed via requirements.txt)
            try:
                _self.nlp = spacy.load("en_core_web_sm")
                st.success("✅ spaCy English model loaded successfully!")
                
            except OSError as e:
                st.warning("⚠️ spaCy model not found in expected location")
                st.info("📋 Trying alternative loading methods...")
                
                # Method 2: Try different model names that might be available
                alternative_models = [
                    "en_core_web_sm",
                    "en",
                    "en_core_web_md",
                    "en_core_web_lg"
                ]
                
                loaded = False
                for model_name in alternative_models:
                    try:
                        st.info(f"🔍 Trying model: {model_name}")
                        _self.nlp = spacy.load(model_name)
                        st.success(f"✅ Loaded alternative model: {model_name}")
                        loaded = True
                        break
                    except:
                        continue
                
                if not loaded:
                    # Method 3: Try installing and loading
                    st.info("📦 Attempting to install spaCy model...")
                    try:
                        import subprocess
                        import sys
                        
                        # Try to install the model
                        result = subprocess.run([
                            sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--user"
                        ], capture_output=True, text=True, timeout=180)
                        
                        if result.returncode == 0:
                            st.info("✅ Installation completed, trying to load...")
                            _self.nlp = spacy.load("en_core_web_sm")
                            st.success("✅ spaCy model installed and loaded!")
                        else:
                            st.error(f"❌ Installation failed: {result.stderr}")
                            raise Exception("Installation failed")
                            
                    except Exception as install_error:
                        st.error(f"❌ Could not install spaCy model: {str(install_error)}")
                        st.info("💡 Using enhanced fallback chunking method")
                        _self.nlp = None
                        
        except Exception as e:
            st.error(f"❌ Unexpected error with spaCy: {str(e)}")
            st.info("💡 Using enhanced fallback chunking method")
            _self.nlp = None
        
        # Load HuggingFace tokenizer
        try:
            # Get HuggingFace token from secrets if available
            hf_token = None
            try:
                hf_token = st.secrets["HF_API_KEY"]
                st.success("✅ HuggingFace token loaded from secrets")
            except KeyError:
                st.warning("⚠️ HF_API_KEY not found in secrets. Some models may not be accessible.")
            
            # Load tokenizer with authentication if token is available
            if hf_token:
                _self.tokenizer = AutoTokenizer.from_pretrained(
                    _self.model_name, 
                    use_auth_token=hf_token
                )
            else:
                _self.tokenizer = AutoTokenizer.from_pretrained(_self.model_name)
            
            if _self.tokenizer.pad_token is None:
                _self.tokenizer.pad_token = _self.tokenizer.eos_token
            
            st.success(f"✅ Successfully loaded tokenizer: {_self.model_name}")
            
        except Exception as e:
            st.warning(f"⚠️ Could not load tokenizer '{_self.model_name}': {str(e)}")
            st.info("💡 Using basic tokenization fallback")
            _self.tokenizer = None
            # Don't mark as failure since we have fallbacks
        
        return True  # Always return True since we have fallback methods
    
    def chunk_text_intelligently(self, text: str) -> List[Dict]:
        """
        Chunk text using spaCy for intelligent paragraph and sentence boundaries
        Returns list of chunk dictionaries with metadata
        """
        if not self.nlp:
            return self._fallback_chunk(text)
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract paragraphs (split by double newlines or more)
        paragraphs = self._extract_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        current_tokens = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Process paragraph with spaCy
            para_doc = self.nlp(paragraph)
            sentences = list(para_doc.sents)
            
            for sent in sentences:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue
                
                # Estimate token count for this sentence
                sentence_tokens = self._estimate_tokens(sentence_text)
                
                # Check if adding this sentence would exceed chunk size
                if (current_tokens + sentence_tokens > config.CHUNK_SIZE and 
                    current_chunk.strip()):
                    
                    # Save current chunk
                    if current_chunk.strip():
                        chunks.append(self._create_chunk_dict(
                            current_chunk.strip(), 
                            current_sentences,
                            para_idx
                        ))
                    
                    # Start new chunk with overlap consideration
                    overlap_sentences = self._get_overlap_sentences(
                        current_sentences, 
                        config.CHUNK_OVERLAP
                    )
                    
                    current_chunk = " ".join(overlap_sentences) + " " + sentence_text
                    current_sentences = overlap_sentences + [sentence_text]
                    current_tokens = self._estimate_tokens(current_chunk)
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence_text
                    else:
                        current_chunk = sentence_text
                    
                    current_sentences.append(sentence_text)
                    current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), 
                current_sentences,
                len(paragraphs) - 1
            ))
        
        return chunks
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        # Split by double newlines or more, but preserve some structure
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove excessive whitespace but preserve single line breaks
            cleaned = re.sub(r'\s+', ' ', para.strip())
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using LLaMA tokenizer or fallback"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except:
                pass
        
        # Fallback: rough estimation (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentences for overlap based on character count"""
        if not sentences:
            return []
        
        overlap_sentences = []
        current_overlap = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if current_overlap + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                current_overlap += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_dict(self, chunk_text: str, sentences: List[str], paragraph_idx: int) -> Dict:
        """Create a structured chunk dictionary with metadata"""
        return {
            'text': chunk_text,
            'sentences': sentences,
            'sentence_count': len(sentences),
            'paragraph_index': paragraph_idx,
            'char_count': len(chunk_text),
            'estimated_tokens': self._estimate_tokens(chunk_text),
            'type': 'paragraph_based'
        }
    
    def _fallback_chunk(self, text: str) -> List[Dict]:
        """Enhanced fallback chunking method when spaCy is not available"""
        st.info("🔄 Using enhanced regex-based sentence chunking")
        
        chunks = []
        
        # Better sentence splitting using regex
        # This pattern handles common sentence endings better
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean sentences
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Filter very short fragments
                clean_sentences.append(sent)
        
        if not clean_sentences:
            return [self._create_chunk_dict(text, [text], 0)]
        
        current_chunk = ""
        current_sentences = []
        para_idx = 0
        
        for i, sentence in enumerate(clean_sentences):
            # Estimate if this sentence would make chunk too long
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > config.CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk_dict(
                    current_chunk.strip(), 
                    current_sentences,
                    para_idx
                ))
                
                # Start new chunk with overlap (last sentence)
                if current_sentences:
                    current_chunk = current_sentences[-1] + " " + sentence
                    current_sentences = [current_sentences[-1], sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
                
                para_idx += 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(), 
                current_sentences,
                para_idx
            ))
        
        st.success(f"✅ Created {len(chunks)} chunks using enhanced fallback method")
        return chunks
    
    def chunk_text_with_llama(self, text: str) -> List[str]:
        """
        Main chunking method - returns list of chunk texts for backward compatibility
        """
        chunk_dicts = self.chunk_text_intelligently(text)
        return [chunk['text'] for chunk in chunk_dicts]
    
    def get_chunk_metadata(self, text: str) -> List[Dict]:
        """
        Get detailed chunking information with metadata
        """
        return self.chunk_text_intelligently(text)


    
 



class RAGSystem:
    """Main RAG System class with incremental processing"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self.chunker = LlamaChunker()
        self.embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL, self.db_path)
        self.llm_manager = LLMManager()
        self.metadata_manager = FileMetadataManager(self.db_path)
        
        # Check if we have existing data
        self.is_ready = len(self.embedding_manager.chunks) > 0
    
    def process_new_documents(self, files) -> Dict:
        """Process only new documents incrementally"""
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
                with st.spinner(f"Processing {file.name} with intelligent chunking..."):
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
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = self.embedding_manager.get_database_stats()
        stats['processed_files'] = self.metadata_manager.get_processed_files()
        return stats
    
    def reset_database(self):
        """Reset the entire database (for testing purposes)"""
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.__init__()  # Reinitialize

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Advanced RAG System - Incremental Document Processing")
    st.markdown("Upload documents and ask questions with real LLaMA chunking and persistent storage!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for configuration and stats
    with st.sidebar:
        st.header("Configuration")
        
        # Model Selection
        st.subheader("LLaMA Model Selection")
        model_options = {
            "microsoft/DialoGPT-medium": "DialoGPT Medium (Default)",
            "microsoft/DialoGPT-large": "DialoGPT Large",
            "distilbert-base-uncased": "DistilBERT (Lightweight)",
            "bert-base-uncased": "BERT Base",
            "facebook/opt-1.3b": "OPT 1.3B",
            "microsoft/codebert-base": "CodeBERT (For Code)",
            "meta-llama/Llama-2-7b-chat-hf": "LLaMA 2 7B (Requires Auth)",
            "custom": "Custom Model"
        }
        
        selected_model = st.selectbox(
            "Choose LLaMA Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        if selected_model == "custom":
            custom_model = st.text_input(
                "Enter custom model name:",
                placeholder="e.g., meta-llama/Llama-2-13b-chat-hf"
            )
            if custom_model:
                selected_model = custom_model
        
        # Update config if different
        if selected_model != config.LLAMA_MODEL:
            config.LLAMA_MODEL = selected_model
            # Reset the system to use new model
            if st.button("Apply Model Change"):
                st.session_state.rag_system = RAGSystem()
                st.success(f"Switched to model: {selected_model}")
                st.rerun()
        
        st.markdown("---")
        
        # API Key Status
        st.subheader("API Keys Status")
        
        # Gemini API Key
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ Gemini API Key loaded")
        except KeyError:
            st.error("❌ Gemini API Key not found")
            st.info("Add GEMINI_API_KEY to secrets")
        
        # HuggingFace API Key
        try:
            hf_key = st.secrets["HF_API_KEY"]
            st.success("✅ HuggingFace Token loaded")
        except KeyError:
            st.warning("⚠️ HF_API_KEY not found")
            st.info("Add HF_API_KEY for LLaMA models")
        
        st.markdown("---")
        
        # System Statistics
        st.header("Database Statistics")
        stats = st.session_state.rag_system.get_system_stats()
        st.metric("Total Chunks", stats['total_chunks'])
        st.metric("Processed Files", len(stats['processed_files']))
        
        if stats['processed_files']:
            with st.expander("Processed Files Details"):
                metadata = st.session_state.rag_system.metadata_manager.load_metadata()
                for file in stats['processed_files']:
                    file_info = metadata.get(file, {})
                    st.write(f"📄 **{file}**")
                    st.write(f"   • Chunks: {file_info.get('chunk_count', 'N/A')}")
                    if 'chunking_method' in file_info:
                        st.write(f"   • Method: {file_info['chunking_method']}")
                        if 'avg_sentences_per_chunk' in file_info:
                            st.write(f"   • Avg sentences/chunk: {file_info['avg_sentences_per_chunk']:.1f}")
                        if 'total_sentences' in file_info:
                            st.write(f"   • Total sentences: {file_info['total_sentences']}")
                        if 'paragraphs_processed' in file_info:
                            st.write(f"   • Paragraphs: {file_info['paragraphs_processed']}")
                    st.write(f"   • Processed: {file_info.get('processed_at', 'N/A')}")
                    st.write("---")
        
        st.markdown("---")
        
        # System information
        st.header("System Info")
        st.info(f"LLM Model: Google Gemini Pro")
        st.info(f"Chunking: spaCy + LLaMA Intelligent")
        st.info(f"Chunk Size: {config.CHUNK_SIZE}")
        st.info(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
        st.info(f"Top K Chunks: {config.TOP_K_CHUNKS}")
        st.info(f"Embedding Model: {config.EMBEDDING_MODEL}")
        
        # Reset button (for development)
        if st.button("🗑️ Reset Database", help="Clear all data and start fresh"):
            st.session_state.rag_system.reset_database()
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files. Only new files will be processed!"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("Process Documents", type="primary"):
                results = st.session_state.rag_system.process_new_documents(uploaded_files)
                
                # Show results
                if results['new_files']:
                    st.success(f"✅ Processed {len(results['new_files'])} new files with {results['new_chunks']} chunks!")
                    for file in results['new_files']:
                        st.write(f"📄 ✅ {file}")
                    
                    # Show chunking details
                    with st.expander("📊 Chunking Details"):
                        metadata = st.session_state.rag_system.metadata_manager.load_metadata()
                        for file in results['new_files']:
                            file_info = metadata.get(file, {})
                            if 'chunking_method' in file_info:
                                st.write(f"**{file}:**")
                                st.write(f"   • Chunks created: {file_info.get('chunk_count', 'N/A')}")
                                st.write(f"   • Total sentences: {file_info.get('total_sentences', 'N/A')}")
                                st.write(f"   • Paragraphs processed: {file_info.get('paragraphs_processed', 'N/A')}")
                                st.write(f"   • Avg sentences per chunk: {file_info.get('avg_sentences_per_chunk', 0):.1f}")
                                st.write("---")
                
                if results['skipped_files']:
                    st.info(f"⏭️ Skipped {len(results['skipped_files'])} already processed files:")
                    for file in results['skipped_files']:
                        st.write(f"📄 ⏭️ {file}")
                
                st.info(f"Total chunks in database: {results['total_chunks']}")
    
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
        "Built with Streamlit • Uses Real LLaMA, Sentence Transformers, FAISS, and Google Gemini"
    )

if __name__ == "__main__":
    main()
