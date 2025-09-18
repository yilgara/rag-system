import hashlib


class DocumentProcessor:
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    def read_pdf(file) -> Tuple[str, str]:
        try:
            file_content = file.read()
            file_hash = DocumentProcessor.get_file_hash(file_content)
            
            # Reset file pointer for PyPDF2
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text, file_hash
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return "", ""
    
    @staticmethod
    def read_txt(file) -> Tuple[str, str]:
        try:
            file_content = file.read()
            file_hash = DocumentProcessor.get_file_hash(file_content)
            text = file_content.decode('utf-8')
            return text, file_hash
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return "", ""
