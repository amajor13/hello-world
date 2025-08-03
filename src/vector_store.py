import os
import logging
import pickle
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

try:
    from langchain.embeddings import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations for document embeddings."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./vector_store",
        embedding_function: Optional[Embeddings] = None,
        embedding_model: str = "text-embedding-ada-002",
        use_openai: bool = False
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the collection in ChromaDB
            persist_directory: Directory to persist the vector store
            embedding_function: Custom embedding function to use
            embedding_model: Model to use for embeddings (if not using custom function)
            use_openai: Whether to use OpenAI embeddings (fallback)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        if embedding_function:
            self.embeddings = embedding_function
            logger.info("✅ Using provided embedding function")
        elif use_openai and OPENAI_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
                logger.info(f"✅ Using OpenAI embeddings: {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                self.embeddings = self._get_fallback_embeddings()
        else:
            self.embeddings = self._get_fallback_embeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize vector store
        self.vector_store = None
        self._load_or_create_vector_store()
    
    def _get_fallback_embeddings(self) -> Embeddings:
        """Get fallback embeddings if OpenAI is not available."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("Using sentence-transformers fallback embeddings")
            return LocalEmbeddings()
        else:
            logger.warning("No embedding models available, using dummy embeddings")
            return DummyEmbeddings()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client=self.client
            )
            logger.info(f"Loaded existing vector store: {self.collection_name}")
        except Exception as e:
            logger.info(f"Creating new vector store: {self.collection_name}")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client=self.client
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info("Documents successfully added and persisted")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter for metadata
            
        Returns:
            List of similar documents
        """
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter for metadata
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"name": self.collection_name, "count": 0, "metadata": {}}
    
    def reset_vector_store(self):
        """Reset the vector store by deleting and recreating it."""
        try:
            self.delete_collection()
            self._load_or_create_vector_store()
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers as fallback for OpenAI."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for local embeddings")
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded local embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            # Try a smaller fallback model
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Loaded fallback embedding model: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return dummy embeddings as ultimate fallback
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 384


class DummyEmbeddings(Embeddings):
    """Dummy embeddings for testing when no real embeddings are available."""
    
    def __init__(self, dimension: int = 384):
        """Initialize dummy embeddings with fixed dimension."""
        self.dimension = dimension
        logger.warning("⚠️  Using dummy embeddings - search quality will be poor")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return dummy embeddings for documents."""
        import hashlib
        embeddings = []
        for text in texts:
            # Create pseudo-embeddings based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to float values between -1 and 1
            embedding = [(b / 128.0 - 1.0) for b in hash_bytes]
            # Pad or truncate to desired dimension
            while len(embedding) < self.dimension:
                embedding.extend(embedding[:self.dimension - len(embedding)])
            embedding = embedding[:self.dimension]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return dummy embedding for query."""
        return self.embed_documents([text])[0]