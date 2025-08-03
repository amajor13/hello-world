import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the RAG chatbot application."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Paths
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./documents")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    
    # Application Settings
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "true").lower() == "true"
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    # Streamlit Configuration
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "RAG Chatbot")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "🤖")
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if cls.USE_OPENAI and not cls.OPENAI_API_KEY:
            print("Warning: USE_OPENAI is True but OPENAI_API_KEY is not set")
            return False
        
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "model": cls.OPENAI_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "use_openai": cls.USE_OPENAI,
            "vector_store_path": cls.VECTOR_STORE_PATH,
            "documents_path": cls.DOCUMENTS_PATH,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "retrieval_k": cls.RETRIEVAL_K
        }