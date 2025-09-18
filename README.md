# RAG System

A Retrieval-Augmented Generation (RAG) system built with Streamlit that allows you to upload documents and ask questions about their content using Google Gemini AI.

---

## Features

- **Document Upload & Processing**: Support for PDF and TXT files with automatic deduplication
- **Chunking**: Uses text chunking for optimal content organization
- **Semantic Search**: FAISS-powered vector database for fast similarity search
- **AI-Powered Answers**: Google Gemini 1.5 Flash integration for natural language responses
- **Persistent Storage**: Maintains processed documents and embeddings across sessions
- **Real-time Statistics**: Track processed files, chunks, and system performance
- **Duplicate Detection**: Automatically skips already processed files using content hashing

---

## Architecture

The system consists of several key components:

- **RAGSystem**: Main orchestrator that coordinates all components
- **Chunker**: Text chunking using spaCy
- **EmbeddingManager**: Handles vector embeddings and FAISS database operations
- **LLMManager**: Manages Google Gemini API interactions
- **FileMetadataManager**: Tracks processed files and prevents duplicates
- **DocumentProcessor**: Handles PDF and text file reading with content hashing

---

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Streamlit Cloud account (for deployment) or local environment

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yilgara/rag-system.git
   cd rag-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Google Gemini API key:**
   
   **For local development:**
   - Create a `.streamlit/secrets.toml` file:
     ```toml
     GEMINI_API_KEY = "your-gemini-api-key-here"
     ```
   
   **For Streamlit Cloud deployment:**
   - Add `GEMINI_API_KEY` to your Streamlit Cloud secrets in the app settings

---

## Usage

### Local Development

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the application:**
   - Open your browser to `http://localhost:8501`

### Using the System

1. **Upload Documents:**
   - Use the file uploader in the left column
   - Select PDF or TXT files (multiple files supported)
   - Click "Process Documents" to add them to the system
   - Already processed files will be automatically skipped

2. **Ask Questions:**
   - Use the question input in the right column
   - Enter natural language questions about your documents
   - View AI-generated answers with relevant source chunks
   - See relevance scores for transparency

3. **Monitor System:**
   - Check the sidebar for database statistics
   - View processed file details and metadata
   - Monitor chunk counts and system performance

---

## Configuration

The system can be configured through `config.py`:

```python
# Example configuration options
CHUNK_SIZE = 1000          # Maximum characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
TOP_K_CHUNKS = 5           # Number of chunks to retrieve
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./vector_db"    # Path to store the database
```
---

## File Structure

```
rag-system/
├── app.py                 # Main Streamlit application
├── rag_system.py         # Core RAG system orchestrator
├── config.py             # Configuration settings
├── chunker.py            # Text chunking logic
├── embedding.py          # Vector embedding management
├── llm.py                # LLM integration (Gemini)
├── file_metadata.py     # File metadata and duplicate tracking
├── document_processor.py # Document reading and processing
├── requirements.txt      # Python dependencies
└── README.md             # This file
```
---

**Deployed Application:**  
You can access the live system at: [https://rag-system-yroucfntmeexvd9cwinsob.streamlit.app](https://rag-system-yroucfntmeexvd9cwinsob.streamlit.app)
