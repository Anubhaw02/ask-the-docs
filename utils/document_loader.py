"""
Document Loader Module
Handles PDF and TXT file loading with OCR support for scanned PDFs
"""

import os
import tempfile
from typing import Optional
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


class DocumentLoader:
    """Loads and extracts text from PDF and TXT files"""
    
    def __init__(self):
        # Set tesseract path for Windows (modify if needed)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def load_txt(self, file_path: str) -> str:
        """
        Load text from a TXT file
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            raise Exception(f"Error loading TXT file: {str(e)}")
    
    def load_pdf(self, file_path: str, use_ocr: bool = False) -> str:
        """
        Load text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            use_ocr: Whether to use OCR for scanned PDFs
            
        Returns:
            Extracted text as string
        """
        text = ""
        
        # First, try normal text extraction
        try:
            text = self._extract_text_pypdf(file_path)
            
            # If extraction yielded very little text, it might be scanned
            if len(text.strip()) < 100:
                print("⚠️ Low text extraction - might be a scanned PDF. Trying pdfplumber...")
                text = self._extract_text_pdfplumber(file_path)
            
            # Still no luck? Use OCR
            if len(text.strip()) < 100 or use_ocr:
                print("🔍 Using OCR to extract text from scanned PDF...")
                text = self._extract_text_ocr(file_path)
                
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
        
        return text
    
    def _extract_text_pypdf(self, file_path: str) -> str:
        """Extract text using PyPDF2"""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber (better for tables)"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_text_ocr(self, file_path: str) -> str:
        """Extract text using OCR (for scanned PDFs)"""
        text = ""
        
        # Convert PDF pages to images
        images = convert_from_path(file_path, dpi=300)
        
        # Perform OCR on each image
        for i, image in enumerate(images):
            print(f"   Processing page {i+1}/{len(images)} with OCR...")
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
        
        return text
    
    def load_document(self, file_path: str, use_ocr: bool = False) -> str:
        """
        Load document (auto-detects PDF or TXT)
        
        Args:
            file_path: Path to the file
            use_ocr: Whether to force OCR for PDFs
            
        Returns:
            Extracted text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.load_pdf(file_path, use_ocr=use_ocr)
        elif file_ext == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")