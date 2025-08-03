#!/usr/bin/env python3
"""
Test script to verify free models are working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.config import Config
        print("✅ Config imported")
        
        from src.free_models import get_free_llm, get_free_embeddings, check_model_availability
        print("✅ Free models imported")
        
        from src.document_loader import DocumentLoader
        print("✅ Document loader imported")
        
        from src.vector_store import VectorStoreManager
        print("✅ Vector store imported")
        
        from src.rag_system import RAGChatbot
        print("✅ RAG system imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_availability():
    """Test model availability."""
    print("\n🔍 Checking model availability...")
    
    try:
        from src.free_models import check_model_availability
        availability = check_model_availability()
        
        print(f"   • Transformers: {'✅' if availability.get('transformers') else '❌'}")
        print(f"   • PyTorch: {'✅' if availability.get('torch') else '❌'}")
        print(f"   • Sentence Transformers: {'✅' if availability.get('sentence_transformers') else '❌'}")
        
        return availability
    except Exception as e:
        print(f"❌ Error checking availability: {e}")
        return {}

def test_simple_llm():
    """Test the simple fallback LLM."""
    print("\n🤖 Testing simple LLM...")
    
    try:
        from src.free_models import SimpleLLM
        
        llm = SimpleLLM()
        response = llm._call("What is Python?")
        print(f"✅ LLM Response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        return False

def test_simple_embeddings():
    """Test simple embeddings."""
    print("\n🔢 Testing simple embeddings...")
    
    try:
        from src.free_models import FreeSentenceTransformerEmbeddings
        
        embeddings = FreeSentenceTransformerEmbeddings()
        test_texts = ["Hello world", "This is a test"]
        embedded = embeddings.embed_documents(test_texts)
        
        print(f"✅ Embedded {len(test_texts)} texts, dimensions: {len(embedded[0])}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return False

def test_config():
    """Test configuration."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from src.config import Config
        
        config_summary = Config.get_config_summary()
        print(f"✅ Configuration loaded:")
        print(f"   • Use OpenAI: {config_summary['use_openai']}")
        print(f"   • Model: {config_summary['model']}")
        print(f"   • Embedding Model: {config_summary['embedding_model']}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing config: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Free RAG Chatbot - Test Suite")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test configuration
    if not test_config():
        success = False
    
    # Test model availability
    availability = test_model_availability()
    
    # Test simple LLM (always works)
    if not test_simple_llm():
        success = False
    
    # Test simple embeddings (always works)
    if not test_simple_embeddings():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: python launch.py")
        print("   2. Or run: python cli_chat.py --load-docs")
        
        if not availability.get('transformers'):
            print("\n💡 Note: For better performance, install transformers:")
            print("   pip install transformers torch sentence-transformers")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)