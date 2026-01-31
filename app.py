"""
Ask The Docs - RAG-based Document Q&A Application
Main Streamlit application
"""

import os
import streamlit as st
from datetime import datetime
from utils.document_loader import DocumentLoader
from utils.text_processor import TextProcessor
from utils.embeddings import EmbeddingManager
from utils.llm_handler import LLMHandler


# Page configuration
st.set_page_config(
    page_title="Ask The Docs",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []


def log_qa(question: str, answer: str):
    """Log question and answer to file"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "qa_history.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")


def process_document(uploaded_file, use_ocr: bool = False):
    """Process uploaded document"""
    
    # Save uploaded file temporarily
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("📄 Loading document..."):
        # Load document
        loader = DocumentLoader()
        text = loader.load_document(file_path, use_ocr=use_ocr)
        
        if len(text.strip()) < 50:
            st.error("❌ Could not extract enough text from document. Try enabling OCR for scanned PDFs.")
            return False
        
        st.success(f"✅ Loaded {len(text)} characters from document")
    
    with st.spinner("✂️ Chunking document..."):
        # Chunk text
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        chunks = processor.chunk_text(text)
        st.success(f"✅ Created {len(chunks)} chunks")
    
    with st.spinner("🔢 Creating embeddings and building index..."):
        # Create embeddings and build FAISS index
        embedding_manager = EmbeddingManager()
        embedding_manager.build_index(chunks)
        st.session_state.embedding_manager = embedding_manager
        st.success("✅ Vector index built successfully")
    
    with st.spinner("🤖 Loading language model..."):
        # Load LLM
        if st.session_state.llm_handler is None:
            llm_handler = LLMHandler()
            st.session_state.llm_handler = llm_handler
        st.success("✅ LLM ready")
    
    st.session_state.document_processed = True
    return True


def main():
    """Main application"""
    
    # Header
    st.title("📚 Ask The Docs")
    st.markdown("### RAG-based Document Question Answering System")
    st.markdown("Upload a PDF or TXT file and ask questions about its content!")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload a document to ask questions about"
        )
        
        use_ocr = st.checkbox(
            "Enable OCR (for scanned PDFs)",
            help="Enable this if your PDF is a scanned image"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"📊 **Size:** {uploaded_file.size / 1024:.2f} KB")
            
            if st.button("🚀 Process Document", type="primary"):
                success = process_document(uploaded_file, use_ocr)
                if success:
                    st.balloons()
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        if st.session_state.document_processed:
            st.success("✅ Document processed")
            st.info(f"💬 Questions asked: {len(st.session_state.qa_history)}")
        else:
            st.warning("⏳ No document processed")
    
    # Main area
    if not st.session_state.document_processed:
        st.info("👈 Please upload and process a document from the sidebar to get started")
        
        # Instructions
        st.markdown("### 📖 How it works:")
        st.markdown("""
        1. **Upload** a PDF or TXT document
        2. **Enable OCR** if your PDF is scanned (image-based)
        3. **Click Process** to analyze the document
        4. **Ask questions** about the content
        5. Get **accurate answers** based only on your document!
        """)
        
    else:
        # Question input
        st.markdown("### 💬 Ask a Question")
        
        question = st.text_input(
            "Your question:",
            placeholder="What is the main topic of this document?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary")
        
        if ask_button and question:
            with st.spinner("🔍 Searching for relevant information..."):
                # Search for relevant chunks
                results = st.session_state.embedding_manager.search(question, top_k=3)
                context_chunks = [chunk for chunk, _ in results]
                
                st.markdown("#### 📝 Retrieved Context:")
                with st.expander("View retrieved chunks"):
                    for i, (chunk, distance) in enumerate(results, 1):
                        st.markdown(f"**Chunk {i}** (distance: {distance:.4f})")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.markdown("---")
            
            with st.spinner("🤔 Generating answer..."):
                # Generate answer
                answer = st.session_state.llm_handler.generate_answer(
                    question, 
                    context_chunks
                )
                
                # Display answer
                st.markdown("#### ✅ Answer:")
                st.success(answer)
                
                # Log Q&A
                log_qa(question, answer)
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now()
                })
        
        # Show Q&A history
        if st.session_state.qa_history:
            st.markdown("---")
            st.markdown("### 📜 Question History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:]), 1):
                with st.expander(f"Q{i}: {qa['question'][:60]}..."):
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown(f"**Answer:** {qa['answer']}")
                    st.caption(f"Asked at: {qa['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()