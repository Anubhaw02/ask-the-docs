"""
LLM Handler Module
Handles answer generation using HuggingFace transformers
"""

from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class LLMHandler:
    """Handles LLM-based answer generation"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize LLM handler
        
        Args:
            model_name: HuggingFace model name
        """
        print(f" Loading LLM model: {model_name}")
        print("   This may take a minute...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Use CPU
        self.device = "cpu"
        self.model.to(self.device)
        
        print(f" LLM model loaded on {self.device}")
    
    def generate_answer(
        self, 
        question: str, 
        context_chunks: List[str],
        max_length: int = 200
    ) -> str:
        """
        Generate answer based on question and context
        
        Args:
            question: User's question
            context_chunks: List of relevant context chunks
            max_length: Maximum length of generated answer
            
        Returns:
            Generated answer as string
        """
        # Combine context chunks
        context = "\n\n".join(context_chunks)
        
        # Create prompt
        prompt = f"""Context information:
{context}

Based on the context above, answer the following question. If the answer cannot be found in the context, say "I cannot find this information in the provided document."

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate
        print(" Generating answer...")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()