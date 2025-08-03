#!/usr/bin/env python3
"""
Quick launch script for the Free RAG Chatbot.
This script handles everything needed to get the chatbot running with free models.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🤖 FREE RAG CHATBOT - QUICK LAUNCH")
    print("=" * 60)
    print("✅ Uses completely FREE open-source models")
    print("✅ No API keys required")
    print("✅ Works offline after initial model download")
    print("=" * 60)


def check_dependencies():
    """Check if dependencies are installed."""
    try:
        import streamlit
        import transformers
        import torch
        import sentence_transformers
        import langchain
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing free model dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "streamlit", "transformers", "torch", "sentence-transformers",
            "langchain", "chromadb", "langchain-community"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Set up environment variables for free models."""
    env_content = """# Free RAG Chatbot Configuration
USE_OPENAI=false
FREE_LLM_MODEL=conversational
FREE_EMBEDDING_MODEL=fast
VECTOR_STORE_PATH=./vector_store
DOCUMENTS_PATH=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
MODEL_DEVICE=auto
MAX_MODEL_LENGTH=512
MODEL_TEMPERATURE=0.7
MAX_CONVERSATION_HISTORY=10
PAGE_TITLE=Free RAG Chatbot
PAGE_ICON=🤖
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("✅ Environment configured for free models")


def create_directories():
    """Create necessary directories."""
    for dir_name in ["documents", "vector_store"]:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Directories created")


def check_sample_documents():
    """Check if sample documents exist."""
    docs_path = Path("documents")
    doc_files = list(docs_path.glob("*.txt")) + list(docs_path.glob("*.md"))
    
    if doc_files:
        print(f"✅ Found {len(doc_files)} sample documents")
        return True
    else:
        print("⚠️  No sample documents found")
        return False


def run_cli_mode():
    """Run the CLI version of the chatbot."""
    print("\n🚀 Launching CLI mode...")
    print("First, let's load the sample documents...")
    
    try:
        # Load documents first
        subprocess.run([sys.executable, "cli_chat.py", "--load-docs"], check=True)
        
        print("\n🎉 Documents loaded! Starting interactive chat...")
        print("💡 Type 'quit' to exit, 'clear' to clear history")
        
        # Start interactive chat
        subprocess.run([sys.executable, "cli_chat.py"], check=False)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running CLI: {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


def run_web_mode():
    """Run the web interface."""
    print("\n🌐 Launching web interface...")
    print("📖 Open your browser to the URL shown below")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web interface: {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


def main():
    """Main launch function."""
    print_banner()
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\n🔧 Installing required dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    create_directories()
    
    # Check for sample documents
    has_docs = check_sample_documents()
    if not has_docs:
        print("📚 Adding sample documents for testing...")
    
    # Ask user which mode to run
    print("\n🎯 Choose launch mode:")
    print("1. 💻 Command Line Interface (CLI)")
    print("2. 🌐 Web Interface (Streamlit)")
    print("3. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_cli_mode()
                break
            elif choice == "2":
                run_web_mode()
                break
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()