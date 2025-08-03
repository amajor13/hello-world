import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .free_models import get_free_llm, get_free_embeddings, check_model_availability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manages conversation history for context-aware responses."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation history.
        
        Args:
            max_history: Maximum number of conversation turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_interaction(self, user_query: str, bot_response: str):
        """Add a new interaction to history."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "bot": bot_response
        }
        self.history.append(interaction)
        
        # Keep only the most recent interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_string(self, include_last_n: int = 3) -> str:
        """Get conversation history as a formatted string."""
        if not self.history:
            return ""
        
        recent_history = self.history[-include_last_n:]
        context_parts = []
        
        for interaction in recent_history:
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['bot']}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear all conversation history."""
        self.history = []


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        print(token, end="", flush=True)


class RAGChatbot:
    """Main RAG chatbot system combining retrieval and generation."""
    
    def __init__(
        self,
        documents_path: str = "./documents",
        vector_store_path: str = "./vector_store",
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        use_openai: bool = False,  # Default to free models
        free_llm_model: str = "conversational",
        free_embedding_model: str = "fast",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retrieval_k: int = 4,
        model_device: str = "auto",
        max_model_length: int = 512,
        model_temperature: float = 0.7
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            documents_path: Path to documents directory
            vector_store_path: Path to vector store
            openai_api_key: OpenAI API key (optional)
            model_name: OpenAI model name
            embedding_model: OpenAI embedding model name
            use_openai: Whether to use OpenAI or free models
            free_llm_model: Type of free LLM model to use
            free_embedding_model: Type of free embedding model to use
            chunk_size: Text chunk size for splitting
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of documents to retrieve
            model_device: Device to run models on
            max_model_length: Maximum model output length
            model_temperature: Model temperature
        """
        self.documents_path = documents_path
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        self.free_llm_model = free_llm_model
        self.free_embedding_model = free_embedding_model
        self.retrieval_k = retrieval_k
        self.model_device = model_device
        self.max_model_length = max_model_length
        self.model_temperature = model_temperature
        
        # Check model availability
        availability = check_model_availability()
        logger.info(f"Model availability: {availability}")
        
        # Set API key if using OpenAI
        if openai_api_key and use_openai:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embeddings
        if use_openai and OPENAI_AVAILABLE and openai_api_key:
            try:
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
                logger.info(f"✅ Using OpenAI embeddings: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
                logger.info("Falling back to free embeddings...")
                self.embeddings = get_free_embeddings(free_embedding_model)
                self.use_openai = False
        else:
            logger.info(f"Using free embeddings: {free_embedding_model}")
            self.embeddings = get_free_embeddings(free_embedding_model)
        
        self.vector_store_manager = VectorStoreManager(
            persist_directory=vector_store_path,
            embedding_function=self.embeddings,
            use_openai=False  # Always use ChromaDB with custom embeddings
        )
        
        self.conversation_history = ConversationHistory()
        
        # Initialize LLM
        if use_openai and OPENAI_AVAILABLE and openai_api_key:
            try:
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=model_temperature,
                    streaming=False
                )
                logger.info(f"✅ Using OpenAI LLM: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
                logger.info("Falling back to free LLM...")
                self.llm = get_free_llm(free_llm_model)
                self.use_openai = False
        else:
            logger.info(f"Using free LLM: {free_llm_model}")
            self.llm = get_free_llm(free_llm_model)
        
        # Initialize prompts
        self._setup_prompts()
        
        # Initialize retrieval chain
        self._setup_retrieval_chain()
    
    def _setup_prompts(self):
        """Setup prompt templates for the chatbot."""
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 

Guidelines:
1. Use the context documents to provide accurate and detailed answers
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite the source documents when possible
4. Maintain a helpful and conversational tone
5. Consider the conversation history for context-aware responses
6. Keep responses concise but informative

Context Documents:
{context}

Conversation History:
{history}

Current Question: {question}

Please provide a helpful answer based on the context provided."""
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])
        
        # Simplified prompt for free models
        self.simple_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context, answer the question concisely and accurately.

Context: {context}

Question: {question}

Answer:"""
        )
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain."""
        try:
            self.retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={"k": self.retrieval_k}
                ),
                return_source_documents=True
            )
        except Exception as e:
            logger.warning(f"Could not setup retrieval chain: {e}")
            self.retrieval_chain = None
    
    def load_documents(self, reset_vector_store: bool = False) -> int:
        """
        Load documents from the documents directory into the vector store.
        
        Args:
            reset_vector_store: Whether to reset the vector store before loading
            
        Returns:
            Number of document chunks loaded
        """
        if reset_vector_store:
            self.vector_store_manager.reset_vector_store()
        
        # Load documents
        documents = self.document_loader.load_documents_from_directory(
            self.documents_path
        )
        
        if documents:
            self.vector_store_manager.add_documents(documents)
            # Reinitialize retrieval chain with new documents
            self._setup_retrieval_chain()
            logger.info(f"Successfully loaded {len(documents)} document chunks")
        else:
            logger.warning("No documents found to load")
        
        return len(documents)
    
    def add_document(self, file_path: str) -> int:
        """
        Add a single document to the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of chunks added
        """
        documents = self.document_loader.load_single_document(file_path)
        
        if documents:
            self.vector_store_manager.add_documents(documents)
            # Reinitialize retrieval chain with new documents
            self._setup_retrieval_chain()
            logger.info(f"Successfully added {len(documents)} chunks from {file_path}")
        
        return len(documents)
    
    def retrieve_relevant_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (defaults to self.retrieval_k)
            
        Returns:
            List of relevant documents
        """
        k = k or self.retrieval_k
        return self.vector_store_manager.similarity_search(query, k=k)
    
    def generate_response(
        self,
        query: str,
        use_history: bool = True,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using RAG.
        
        Args:
            query: User query
            use_history: Whether to include conversation history
            include_sources: Whether to include source documents in response
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(query)
            
            if not relevant_docs:
                response_text = "I don't have any relevant information in my knowledge base to answer your question. Please try rephrasing your question or check if the relevant documents have been loaded."
                return {
                    "response": response_text,
                    "sources": [],
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare context
            context = "\n\n".join([
                f"Source: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            # Truncate context if too long for free models
            if not self.use_openai and len(context) > 2000:
                context = context[:2000] + "..."
            
            # Get conversation history if requested
            history = ""
            if use_history and self.use_openai:
                history = self.conversation_history.get_context_string()
            
            # Generate response using the appropriate method
            if self.use_openai and hasattr(self.llm, 'predict_messages'):
                # For OpenAI chat models
                messages = [
                    SystemMessage(content=self.system_prompt.format(
                        context=context,
                        history=history,
                        question=query
                    )),
                    HumanMessage(content=query)
                ]
                response = self.llm.predict_messages(messages)
                response_text = response.content
            else:
                # For free models - use simpler prompt
                prompt = self.simple_prompt.format(
                    context=context,
                    question=query
                )
                response_text = self.llm.predict(prompt)
            
            # Clean up response
            response_text = response_text.strip()
            if not response_text:
                response_text = "I understand your question, but I need more context to provide a helpful answer."
            
            # Prepare sources information
            sources = []
            if include_sources:
                for doc in relevant_docs:
                    sources.append({
                        "filename": doc.metadata.get('filename', 'Unknown'),
                        "source": doc.metadata.get('source', 'Unknown'),
                        "chunk_index": doc.metadata.get('chunk_index', 0),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            # Add to conversation history
            self.conversation_history.add_interaction(query, response_text)
            
            return {
                "response": response_text,
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "num_sources": len(relevant_docs),
                "model_type": "OpenAI" if self.use_openai else "Free"
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_response = f"I encountered an error while processing your question. Please try rephrasing your question or try again."
            return {
                "response": error_response,
                "sources": [],
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def chat(self, query: str) -> str:
        """
        Simple chat interface that returns just the response text.
        
        Args:
            query: User query
            
        Returns:
            Response text
        """
        result = self.generate_response(query)
        return result["response"]
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        return self.vector_store_manager.get_collection_info()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system configuration."""
        availability = check_model_availability()
        return {
            "model_type": "OpenAI" if self.use_openai else "Free",
            "llm_model": self.model_name if self.use_openai else self.free_llm_model,
            "embedding_model": self.embedding_model if self.use_openai else self.free_embedding_model,
            "device": self.model_device,
            "availability": availability,
            "vector_store": self.get_vector_store_info()
        }
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear_history()
    
    def reset_system(self):
        """Reset the entire system (vector store and conversation history)."""
        self.vector_store_manager.reset_vector_store()
        self.conversation_history.clear_history()
        logger.info("System reset completed")