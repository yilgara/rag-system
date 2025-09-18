import streamlit as st


from config import config
from rag_system import RAGSystem



def main():
    st.set_page_config(
        page_title="RAG System",
        layout="wide"
    )
    
    st.title("RAG System")
    st.markdown("Upload documents and ask questions with persistent storage!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for configuration and stats
    with st.sidebar:
        
        
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
        st.info(f"LLM Model: Google Gemini 1.5 Flash")
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
                    for file in results['new_files']:
                        st.success(f"Processed new file: 📄 **{file}**")
                    
                
                if results['skipped_files']:
                    for file in results['skipped_files']:
                        st.info(f"Skipped already processed file: 📄 **{file}**")

                st.session_state.uploaded_files = []
         
    
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
        "Built with Streamlit • Uses Sentence Transformers, FAISS, and Google Gemini"
    )

if __name__ == "__main__":
    main()
