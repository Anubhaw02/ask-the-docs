"""
Embeddings Module
Handles embedding generation and FAISS vector store management
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingManager:
    """Manages embeddings and FAISS index"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        print(f"✅ Model loaded (embedding dimension: {self.dimension})")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        print(f"🔢 Creating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[str]):
        """
        Build FAISS index from text chunks
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")
        
        self.chunks = chunks
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most relevant chunks
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_text, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results