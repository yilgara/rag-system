import streamlit as st
from config import config
import spacy
import re



class Chunker:
    
    def __init__(self):
        self.nlp = None
        self.load_models()


    def load_models(self):
        self.nlp = self._load_models_cached()

    
    @st.cache_resource
    def _load_models_cached(_self):
        nlp_model = None
    
        # Load spaCy model
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except Exception as e:
            pass
    
        return nlp_model
    
        
    def chunk_text(self, text):
        if not self.nlp:
            return None
            
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
    
    def _extract_paragraphs(self, text):

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



    
    def _estimate_tokens(self, text):
        #rough estimation (1 token ≈ 4 characters for English)
        return len(text) // 4


    
    def _get_overlap_sentences(self, sentences, overlap_chars):
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


    
    def _create_chunk_dict(self, chunk_text, sentences, paragraph_idx):
        return {
            'text': chunk_text,
            'sentences': sentences,
            'sentence_count': len(sentences),
            'paragraph_index': paragraph_idx,
            'char_count': len(chunk_text),
            'estimated_tokens': self._estimate_tokens(chunk_text),
            'type': 'paragraph_based'
        }
    
   
