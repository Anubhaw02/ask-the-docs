# 📚 Ask The Docs - RAG-based Document Q&A System

A production-ready document question-answering application built with Retrieval Augmented Generation (RAG). Upload PDF or TXT files and ask questions about their content using AI.

## Features

- **Multi-format Support**: PDF and TXT files
- **OCR Integration**: Handles scanned PDFs using Tesseract
- **RAG Pipeline**: Retrieval Augmented Generation for accurate answers
- **Vector Search**: FAISS for efficient similarity search
- **Open-Source LLM**: Uses Google FLAN-T5 (no API keys needed)
- **Question History**: Logs all Q&A interactions
- **Dockerized**: Easy deployment anywhere
- **AWS Ready**: Deployment instructions for EC2

## Architecture
```
User Upload (PDF/TXT)
    ↓
Document Loader (with OCR)
    ↓
Text Chunking (500 chars, 50 overlap)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
FAISS Vector Store
    ↓
User Question → Similarity Search → Top 3 Chunks
    ↓
LLM (FLAN-T5) → Answer Generation
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI** | Streamlit | Web interface |
| **PDF Processing** | PyPDF2, pdfplumber, pdf2image | Text extraction |
| **OCR** | Tesseract | Scanned PDF handling |
| **Text Chunking** | LangChain | Smart text splitting |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Vector representations |
| **Vector DB** | FAISS | Similarity search |
| **LLM** | HuggingFace Transformers (FLAN-T5-base) | Answer generation |
| **Containerization** | Docker | Deployment |
| **Cloud** | AWS EC2 | Hosting |

## Prerequisites

- Python 3.9+
- Docker (for containerization)
- Tesseract OCR
- 4GB+ RAM (for running models)

## Quick Start - Local Development

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd ask-the-docs
```

### 2. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr poppler-utils -y
```

**Mac:**
```bash
brew install tesseract poppler
```

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH

### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Application
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t ask-the-docs:latest .
```

### Run Container
```bash
docker run -p 8501:8501 ask-the-docs:latest
```

Access at: `http://localhost:8501`

## ☁️ AWS EC2 Deployment

### 1. Launch EC2 Instance

- **AMI**: Ubuntu Server 22.04 LTS
- **Instance Type**: t2.medium (4GB RAM required)
- **Security Group**: Allow ports 22 (SSH), 80 (HTTP), 8501 (Streamlit)
- **Storage**: 30GB

### 2. Connect to EC2
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 3. Install Docker
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

### 4. Transfer and Build
```bash
# On local machine
tar -czf ask-the-docs.tar.gz .
scp -i your-key.pem ask-the-docs.tar.gz ubuntu@your-ec2-ip:~

# On EC2
mkdir ask-the-docs
tar -xzf ask-the-docs.tar.gz -C ask-the-docs/
cd ask-the-docs
docker build -t ask-the-docs:latest .
```

### 5. Run Application
```bash
docker run -d -p 8501:8501 --name ask-the-docs-app ask-the-docs:latest
```

Access at: `http://your-ec2-ip:8501`

## 📁 Project Structure
```
ask-the-docs/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # This file
├── utils/
│   ├── document_loader.py     # PDF/TXT loading with OCR
│   ├── text_processor.py      # Text chunking
│   ├── embeddings.py          # Vector embeddings & FAISS
│   └── llm_handler.py         # LLM answer generation
├── data/uploads/              # Uploaded files (runtime)
└── logs/qa_history.txt        # Q&A logs
```

## Functioning:

1. **Document Upload**: User uploads PDF or TXT file
2. **Text Extraction**: PyPDF2/pdfplumber extract text; OCR handles scanned PDFs
3. **Chunking**: Text split into 500-character chunks with 50-character overlap
4. **Embedding**: Each chunk converted to 384-dimensional vector using MiniLM
5. **Indexing**: Vectors stored in FAISS index for fast retrieval
6. **Question Processing**: User question converted to same vector space
7. **Retrieval**: Top 3 most similar chunks retrieved from FAISS
8. **Answer Generation**: FLAN-T5 generates answer based on retrieved context
9. **Logging**: Q&A pair logged to history file

## Models Used

### Embedding Model: `all-MiniLM-L6-v2`
- **Size**: 80MB
- **Dimension**: 384
- **Speed**: ~0.1s per chunk on CPU
- **Quality**: State-of-the-art for sentence embeddings

### LLM: `google/flan-t5-base`
- **Size**: 850MB
- **Parameters**: 250M
- **Speed**: 2-5s per answer on CPU
- **Strength**: Question answering, summarization

## Limitations

1. **Performance**: 
   - LLM inference on CPU is slow (2-5s per answer)
   - OCR processing can take time for large scanned PDFs

2. **Accuracy**:
   - Answers limited to document content
   - OCR may have errors on poor-quality scans
   - Model may struggle with complex questions

3. **Scale**:
   - Designed for single-user use
   - No multi-user session management
   - Limited to documents < 100 pages for good performance

4. **Resource Requirements**:
   - Needs 4GB RAM minimum
   - Models take ~1GB disk space

## Troubleshooting

### "Low text extraction" warning
- Your PDF might be scanned. Enable OCR checkbox.

### Out of memory errors
- Use t2.medium or larger instance (4GB+ RAM)
- Reduce chunk size in `text_processor.py`

### Slow responses
- Normal on CPU. GPU would be 10x faster but costs more.
- Consider using smaller LLM like `flan-t5-small`

### Docker build fails
- Ensure you have 10GB free disk space
- Check internet connection (downloads ~2GB)

## Performance Metrics

- **Document Processing**: ~30s for 50-page PDF
- **OCR Processing**: ~5s per page
- **Embedding Creation**: ~0.1s per chunk
- **Answer Generation**: 2-5s on CPU
- **Memory Usage**: ~3GB RAM

## Future Improvements

- [ ] GPU support for faster inference
- [ ] Multiple document support
- [ ] Conversational context (multi-turn Q&A)
- [ ] Support for more file types (DOCX, HTML)
- [ ] User authentication
- [ ] API endpoint for programmatic access
- [ ] Fine-tuned models for specific domains

## License

MIT License - Feel free to use for learning and interviews!

## Author

Built as an interview project demonstrating:
- RAG implementation
- LLMOps best practices
- Docker containerization
- Cloud deployment
- Production-ready code structure

## Acknowledgments

- HuggingFace for transformer models
- Facebook AI for FAISS
- LangChain for chunking utilities
- Streamlit for rapid UI development
