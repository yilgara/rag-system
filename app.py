import streamlit as st


from config import config
from rag_system import RAGSystem



def main():
    st.set_page_config(
        page_title="RAG System",
        layout="wide"
    )
    
    st.title("RAG System")
    st.markdown("Upload documents and ask questions with LLaMA chunking and persistent storage!")
    
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
        st.info(f"Chunk Size: {config.CHUNK_SIZE}")
        st.info(f"Chunk Overlap: {config.CHUNK_OVERLAP}")
        st.info(f"Top K Chunks: {config.TOP_K_CHUNKS}")
        st.info(f"Embedding Model: {config.EMBEDDING_MODEL}")
        st.info(f"Tokenizer: {config.LLAMA_MODEL}")
        
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
        "Built with Streamlit • Uses LLaMA, Sentence Transformers, FAISS, and Google Gemini"
    )

if __name__ == "__main__":
    main()
