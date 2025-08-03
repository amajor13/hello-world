import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the RAG chatbot application."""
    
    # OpenAI Configuration (optional)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Free Model Configuration (default)
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "false").lower() == "true"
    FREE_LLM_MODEL: str = os.getenv("FREE_LLM_MODEL", "conversational")  # conversational, general, small, medium
    FREE_EMBEDDING_MODEL: str = os.getenv("FREE_EMBEDDING_MODEL", "fast")  # fast, quality, balanced
    
    # Paths
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./documents")
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    
    # Application Settings
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    # Streamlit Configuration
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "Free RAG Chatbot")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "🤖")
    
    # Model Performance Settings
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # auto, cpu, cuda
    MAX_MODEL_LENGTH: int = int(os.getenv("MAX_MODEL_LENGTH", "512"))
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if cls.USE_OPENAI and not cls.OPENAI_API_KEY:
            print("Warning: USE_OPENAI is True but OPENAI_API_KEY is not set")
            print("Falling back to free models...")
            cls.USE_OPENAI = False
            return True
        
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "use_openai": cls.USE_OPENAI,
            "openai_configured": bool(cls.OPENAI_API_KEY) if cls.USE_OPENAI else False,
            "model": cls.OPENAI_MODEL if cls.USE_OPENAI else f"Free Model ({cls.FREE_LLM_MODEL})",
            "embedding_model": cls.EMBEDDING_MODEL if cls.USE_OPENAI else f"Free Embeddings ({cls.FREE_EMBEDDING_MODEL})",
            "vector_store_path": cls.VECTOR_STORE_PATH,
            "documents_path": cls.DOCUMENTS_PATH,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "retrieval_k": cls.RETRIEVAL_K,
            "device": cls.MODEL_DEVICE,
            "max_length": cls.MAX_MODEL_LENGTH,
            "temperature": cls.MODEL_TEMPERATURE
        }
    
    @classmethod
    def get_model_info(cls) -> dict:
        """Get information about available models."""
        free_models = {
            "llm_models": {
                "conversational": "Microsoft DialoGPT (Small) - Good for conversations",
                "general": "DistilGPT-2 - General purpose, fast",
                "small": "DistilGPT-2 - Smallest, fastest",
                "medium": "Microsoft DialoGPT (Medium) - Better quality, slower"
            },
            "embedding_models": {
                "fast": "all-MiniLM-L6-v2 - Fast and efficient",
                "quality": "all-mpnet-base-v2 - Higher quality",
                "balanced": "paraphrase-MiniLM-L6-v2 - Good balance"
            }
        }
        
        return {
            "free_models": free_models,
            "current_llm": cls.FREE_LLM_MODEL,
            "current_embeddings": cls.FREE_EMBEDDING_MODEL,
            "using_openai": cls.USE_OPENAI
        }