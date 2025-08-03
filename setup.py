#!/usr/bin/env python3
"""
Setup script for the RAG Chatbot project.
Handles installation, dependency setup, and initial configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"📦 {description}")
    try:
        result = subprocess.run(
            command.split(),
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    return True


def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🐍 Creating virtual environment...")
    return run_command("python -m venv venv", "Creating virtual environment")


def get_pip_command():
    """Get the appropriate pip command for the current platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\pip"
    else:
        return "venv/bin/pip"


def install_requirements():
    """Install required packages."""
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    return run_command(
        f"{pip_cmd} install -r requirements.txt",
        "Installing requirements"
    )


def create_directories():
    """Create necessary directories."""
    directories = ["documents", "vector_store", "logs"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory already exists: {dir_name}")


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_path = Path(".env")
    
    if env_path.exists():
        print("✅ .env file already exists")
        return
    
    print("📝 Creating .env file...")
    
    # Get OpenAI API key from user
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    env_content = f"""# RAG Chatbot Configuration
OPENAI_API_KEY={api_key}
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_STORE_PATH=./vector_store
DOCUMENTS_PATH=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
USE_OPENAI=true
MAX_CONVERSATION_HISTORY=10
PAGE_TITLE=RAG Chatbot
PAGE_ICON=🤖
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    if not api_key:
        print("⚠️  Remember to add your OpenAI API key to the .env file")


def run_tests():
    """Run basic tests to verify the setup."""
    print("\n🧪 Running setup verification tests...")
    
    try:
        # Test importing main modules
        sys.path.append(".")
        
        from src.config import Config
        print("✅ Configuration module imported successfully")
        
        from src.document_loader import DocumentLoader
        print("✅ Document loader module imported successfully")
        
        from src.vector_store import VectorStoreManager
        print("✅ Vector store module imported successfully")
        
        from src.rag_system import RAGChatbot
        print("✅ RAG system module imported successfully")
        
        print("✅ All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎉 RAG CHATBOT SETUP COMPLETED!")
    print("="*60)
    
    print("\n📚 NEXT STEPS:")
    print("1. Add your documents to the 'documents' folder")
    print("2. Set your OpenAI API key in the .env file (if not done already)")
    print("3. Run the chatbot!")
    
    print("\n🚀 USAGE:")
    
    # Activation command based on platform
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"""
Web Interface:
   {activate_cmd}
   streamlit run app.py

Command Line Interface:
   {activate_cmd}
   python cli_chat.py --help
   python cli_chat.py --load-docs  # Load documents
   python cli_chat.py             # Start interactive chat

Examples:
   python cli_chat.py --query "What is machine learning?"
   python cli_chat.py --load-docs --reset  # Reset and reload documents
   python cli_chat.py --status            # Show system status
""")
    
    print("\n📖 DOCUMENTATION:")
    print("   • README.md - Project overview and setup instructions")
    print("   • .env.example - Environment variables template")
    print("   • documents/ - Add your PDF, DOCX, TXT, or MD files here")
    
    print("\n💡 TIPS:")
    print("   • The system works better with OpenAI API key")
    print("   • You can use local models by setting USE_OPENAI=false in .env")
    print("   • Add more documents anytime and reload them")
    
    print("="*60)


def main():
    """Main setup function."""
    print("🤖 RAG Chatbot Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Run tests
    if not run_tests():
        print("❌ Setup verification failed")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()


if __name__ == "__main__":
    main()