import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
import time
from typing import List, Dict, Any

# Import our custom modules
from src.config import Config
from src.rag_system import RAGChatbot
from src.document_loader import DocumentLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1e88e5;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1e88e5;
    }
    
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    .stats-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    if "vector_store_info" not in st.session_state:
        st.session_state.vector_store_info = {}

def initialize_chatbot():
    """Initialize the RAG chatbot."""
    if st.session_state.chatbot is None:
        try:
            with st.spinner("Initializing chatbot system..."):
                st.session_state.chatbot = RAGChatbot(
                    documents_path=Config.DOCUMENTS_PATH,
                    vector_store_path=Config.VECTOR_STORE_PATH,
                    openai_api_key=Config.OPENAI_API_KEY,
                    model_name=Config.OPENAI_MODEL,
                    embedding_model=Config.EMBEDDING_MODEL,
                    use_openai=Config.USE_OPENAI,
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP,
                    retrieval_k=Config.RETRIEVAL_K
                )
                
                # Get vector store info
                st.session_state.vector_store_info = st.session_state.chatbot.get_vector_store_info()
                
            st.success("✅ Chatbot initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {str(e)}")
            logger.error(f"Chatbot initialization error: {e}")
            return False
    return True

def load_documents_ui():
    """UI for loading documents."""
    st.subheader("📚 Document Management")
    
    # Check if documents directory exists
    docs_path = Path(Config.DOCUMENTS_PATH)
    if not docs_path.exists():
        st.warning(f"Documents directory '{Config.DOCUMENTS_PATH}' does not exist.")
        if st.button("Create Documents Directory"):
            docs_path.mkdir(parents=True, exist_ok=True)
            st.success(f"Created directory: {Config.DOCUMENTS_PATH}")
            st.rerun()
        return
    
    # Show current documents
    doc_files = list(docs_path.rglob('*'))
    doc_files = [f for f in doc_files if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']]
    
    if doc_files:
        st.write(f"**Found {len(doc_files)} documents:**")
        for doc_file in doc_files:
            st.write(f"• {doc_file.name} ({doc_file.suffix})")
    else:
        st.info("No supported documents found. Add PDF, DOCX, TXT, or MD files to the documents directory.")
    
    # Load documents button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load Documents", type="primary"):
            if st.session_state.chatbot:
                try:
                    with st.spinner("Loading documents into vector store..."):
                        num_chunks = st.session_state.chatbot.load_documents(reset_vector_store=False)
                    
                    if num_chunks > 0:
                        st.success(f"✅ Successfully loaded {num_chunks} document chunks!")
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_info = st.session_state.chatbot.get_vector_store_info()
                    else:
                        st.warning("No documents were loaded. Check if there are supported files in the documents directory.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading documents: {str(e)}")
            else:
                st.error("Please initialize the chatbot first.")
    
    with col2:
        if st.button("🔄 Reset Vector Store", type="secondary"):
            if st.session_state.chatbot:
                try:
                    with st.spinner("Resetting vector store..."):
                        st.session_state.chatbot.reset_system()
                    st.success("✅ Vector store reset successfully!")
                    st.session_state.documents_loaded = False
                    st.session_state.vector_store_info = {}
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"❌ Error resetting vector store: {str(e)}")
            else:
                st.error("Please initialize the chatbot first.")

def file_upload_ui():
    """UI for uploading individual files."""
    st.subheader("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Upload a PDF, DOCX, TXT, or MD file to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            if st.session_state.chatbot:
                try:
                    # Save uploaded file
                    docs_path = Path(Config.DOCUMENTS_PATH)
                    docs_path.mkdir(parents=True, exist_ok=True)
                    file_path = docs_path / uploaded_file.name
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        num_chunks = st.session_state.chatbot.add_document(str(file_path))
                    
                    if num_chunks > 0:
                        st.success(f"✅ Successfully processed {uploaded_file.name} into {num_chunks} chunks!")
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store_info = st.session_state.chatbot.get_vector_store_info()
                    else:
                        st.warning("No content was extracted from the file.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            else:
                st.error("Please initialize the chatbot first.")

def chat_interface():
    """Main chat interface."""
    st.subheader("💬 Chat with your Documents")
    
    # Check if documents are loaded
    if not st.session_state.vector_store_info.get('count', 0):
        st.warning("⚠️ No documents loaded in the vector store. Please load some documents first.")
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📖 Sources ({len(message['sources'])} documents)", expanded=False):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>📄 {source['filename']}</strong> (Chunk {source['chunk_index']})<br>
                                <em>{source['content_preview']}</em>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        if st.session_state.chatbot:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate response
            try:
                with st.spinner("Thinking..."):
                    response_data = st.session_state.chatbot.generate_response(
                        user_input,
                        use_history=True,
                        include_sources=True
                    )
                
                # Add bot response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "sources": response_data.get("sources", []),
                    "timestamp": response_data["timestamp"]
                })
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error(f"Response generation error: {e}")
        else:
            st.error("Please initialize the chatbot first.")

def system_status_ui():
    """Display system status and statistics."""
    st.subheader("📊 System Status")
    
    # Configuration status
    config_valid = Config.validate_config()
    config_status = "✅ Valid" if config_valid else "❌ Invalid"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <strong>Configuration:</strong> {config_status}<br>
            <strong>OpenAI API:</strong> {'✅ Configured' if Config.OPENAI_API_KEY else '❌ Not configured'}<br>
            <strong>Model:</strong> {Config.OPENAI_MODEL}<br>
            <strong>Using OpenAI:</strong> {'Yes' if Config.USE_OPENAI else 'No (Local models)'}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        vector_count = st.session_state.vector_store_info.get('count', 0)
        st.markdown(f"""
        <div class="stats-box">
            <strong>Vector Store:</strong> {'✅ Active' if vector_count > 0 else '❌ Empty'}<br>
            <strong>Document Chunks:</strong> {vector_count}<br>
            <strong>Chat History:</strong> {len(st.session_state.chat_history)} messages<br>
            <strong>Retrieval K:</strong> {Config.RETRIEVAL_K}
        </div>
        """, unsafe_allow_html=True)

def sidebar_ui():
    """Sidebar with controls and settings."""
    with st.sidebar:
        st.header("🤖 RAG Chatbot")
        
        # Initialize button
        if st.button("🚀 Initialize Chatbot", type="primary"):
            initialize_chatbot()
        
        st.divider()
        
        # System status
        system_status_ui()
        
        st.divider()
        
        # Document management
        load_documents_ui()
        
        st.divider()
        
        # File upload
        file_upload_ui()
        
        st.divider()
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_conversation_history()
            st.success("Chat history cleared!")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        with st.expander("Configuration Details"):
            config_summary = Config.get_config_summary()
            for key, value in config_summary.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    sidebar_ui()
    
    # Main content area
    if not Config.validate_config():
        st.error("❌ Configuration is invalid. Please check your environment variables.")
        st.info("Make sure to set your OPENAI_API_KEY if using OpenAI models.")
        return
    
    # Auto-initialize if not already done
    if st.session_state.chatbot is None:
        st.info("👋 Welcome! Please initialize the chatbot using the button in the sidebar.")
    else:
        # Main chat interface
        chat_interface()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <small>
                🤖 Powered by LangChain, OpenAI, and Streamlit | 
                📚 Upload documents and start chatting!
            </small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()