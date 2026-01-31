"""
Text Processor Module
Handles text chunking with overlap for better context retrieval
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextProcessor:
    """Chunks text into smaller pieces for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Cannot chunk empty text")
        
        chunks = self.text_splitter.split_text(text)
        
        # Remove empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        print(f"✅ Created {len(chunks)} chunks from document")
        
        return chunks