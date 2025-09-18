import google.generativeai as genai

class LLMManager:
    
    def __init__(self):
        self.model = None
    
    def initialize_gemini(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            return False
    
    def generate_answer(self, query, context_chunks):
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
