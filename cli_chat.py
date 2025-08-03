#!/usr/bin/env python3
"""
Command-line interface for the RAG Chatbot.
Provides a simple terminal-based interface to interact with the chatbot.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.rag_system import RAGChatbot


class CLIChatInterface:
    """Command-line interface for the RAG chatbot."""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_openai: bool = False):
        """
        Initialize the CLI chat interface.
        
        Args:
            openai_api_key: OpenAI API key (optional)
            use_openai: Whether to use OpenAI models
        """
        self.chatbot = None
        self.use_openai = use_openai
        
        # Set API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    def initialize_chatbot(self) -> bool:
        """Initialize the RAG chatbot."""
        try:
            print("🤖 Initializing RAG Chatbot...")
            
            self.chatbot = RAGChatbot(
                documents_path=Config.DOCUMENTS_PATH,
                vector_store_path=Config.VECTOR_STORE_PATH,
                openai_api_key=Config.OPENAI_API_KEY,
                model_name=Config.OPENAI_MODEL,
                embedding_model=Config.EMBEDDING_MODEL,
                use_openai=self.use_openai or Config.USE_OPENAI,
                free_llm_model=Config.FREE_LLM_MODEL,
                free_embedding_model=Config.FREE_EMBEDDING_MODEL,
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                retrieval_k=Config.RETRIEVAL_K,
                model_device=Config.MODEL_DEVICE,
                max_model_length=Config.MAX_MODEL_LENGTH,
                model_temperature=Config.MODEL_TEMPERATURE
            )
            
            print("✅ Chatbot initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing chatbot: {str(e)}")
            return False
    
    def load_documents(self, reset: bool = False) -> bool:
        """Load documents into the vector store."""
        if not self.chatbot:
            print("❌ Chatbot not initialized")
            return False
        
        try:
            print(f"📚 Loading documents from {Config.DOCUMENTS_PATH}...")
            
            num_chunks = self.chatbot.load_documents(reset_vector_store=reset)
            
            if num_chunks > 0:
                print(f"✅ Successfully loaded {num_chunks} document chunks!")
                return True
            else:
                print("⚠️  No documents found to load")
                return False
                
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return False
    
    def show_status(self):
        """Display system status information."""
        print("\n" + "="*50)
        print("📊 SYSTEM STATUS")
        print("="*50)
        
        # Configuration
        print(f"🔧 Configuration:")
        if self.use_openai or Config.USE_OPENAI:
            print(f"   • Model Type: OpenAI")
            print(f"   • API Key: {'✅ Set' if Config.OPENAI_API_KEY else '❌ Not set'}")
            print(f"   • LLM Model: {Config.OPENAI_MODEL}")
            print(f"   • Embedding Model: {Config.EMBEDDING_MODEL}")
        else:
            print(f"   • Model Type: Free/Open Source")
            print(f"   • LLM Model: {Config.FREE_LLM_MODEL}")
            print(f"   • Embedding Model: {Config.FREE_EMBEDDING_MODEL}")
        
        print(f"   • Documents Path: {Config.DOCUMENTS_PATH}")
        print(f"   • Vector Store Path: {Config.VECTOR_STORE_PATH}")
        print(f"   • Device: {Config.MODEL_DEVICE}")
        
        # Vector Store Info
        if self.chatbot:
            vector_info = self.chatbot.get_vector_store_info()
            doc_count = vector_info.get('count', 0)
            print(f"\n📚 Vector Store:")
            print(f"   • Document Chunks: {doc_count}")
            print(f"   • Status: {'✅ Active' if doc_count > 0 else '❌ Empty'}")
            
            # Model availability
            system_info = self.chatbot.get_system_info()
            availability = system_info.get('availability', {})
            print(f"\n🔧 Model Availability:")
            print(f"   • Transformers: {'✅' if availability.get('transformers') else '❌'}")
            print(f"   • PyTorch: {'✅' if availability.get('torch') else '❌'}")
            print(f"   • Sentence Transformers: {'✅' if availability.get('sentence_transformers') else '❌'}")
        else:
            print(f"\n📚 Vector Store: Not initialized")
        
        print("="*50 + "\n")
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        if not self.chatbot:
            print("❌ Chatbot not initialized")
            return
        
        # Check if documents are loaded
        vector_info = self.chatbot.get_vector_store_info()
        if vector_info.get('count', 0) == 0:
            print("⚠️  No documents loaded in the vector store.")
            print("   Use '--load-docs' to load documents first.")
            return
        
        print("\n" + "="*50)
        print("💬 INTERACTIVE CHAT")
        print("="*50)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'clear' to clear conversation history")
        print("Type 'status' to show system status")
        print("-"*50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                # Check for special commands
                if user_input.lower() == 'clear':
                    self.chatbot.clear_conversation_history()
                    print("🗑️  Conversation history cleared!")
                    continue
                
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Generate response
                print("\n🤖 Assistant: ", end="", flush=True)
                
                response_data = self.chatbot.generate_response(
                    user_input,
                    use_history=True,
                    include_sources=True
                )
                
                print(response_data["response"])
                
                # Show sources if available
                sources = response_data.get("sources", [])
                if sources:
                    print(f"\n📖 Sources ({len(sources)} documents):")
                    for i, source in enumerate(sources, 1):
                        filename = source.get('filename', 'Unknown')
                        chunk_idx = source.get('chunk_index', 0)
                        print(f"   {i}. {filename} (chunk {chunk_idx})")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def single_query(self, query: str):
        """Process a single query and return the response."""
        if not self.chatbot:
            print("❌ Chatbot not initialized")
            return
        
        try:
            print(f"🧑 Query: {query}")
            print("\n🤖 Response:")
            
            response_data = self.chatbot.generate_response(
                query,
                use_history=False,
                include_sources=True
            )
            
            print(response_data["response"])
            
            # Show sources
            sources = response_data.get("sources", [])
            if sources:
                print(f"\n📖 Sources:")
                for source in sources:
                    filename = source.get('filename', 'Unknown')
                    chunk_idx = source.get('chunk_index', 0)
                    print(f"   • {filename} (chunk {chunk_idx})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    """Main function to handle command-line arguments and run the CLI."""
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (can also be set via OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI models (requires API key)"
    )
    
    parser.add_argument(
        "--load-docs",
        action="store_true",
        help="Load documents before starting chat"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before loading documents"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode (non-interactive)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status and exit"
    )
    
    args = parser.parse_args()
    
    # Create CLI interface
    use_openai = args.openai or (args.api_key is not None)
    cli = CLIChatInterface(
        openai_api_key=args.api_key,
        use_openai=use_openai
    )
    
    # Initialize chatbot
    if not cli.initialize_chatbot():
        sys.exit(1)
    
    # Show status if requested
    if args.status:
        cli.show_status()
        return
    
    # Load documents if requested
    if args.load_docs:
        cli.load_documents(reset=args.reset)
    
    # Handle single query mode
    if args.query:
        cli.single_query(args.query)
        return
    
    # Show status and start interactive chat
    cli.show_status()
    cli.interactive_chat()


if __name__ == "__main__":
    main()