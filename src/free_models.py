import os
import logging
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        AutoModel
    )
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLM):
    """Free Hugging Face LLM implementation."""
    
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    temperature: float = 0.7
    device: str = "auto"
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 512,
        temperature: float = 0.7,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            device: Device to run the model on
        """
        super().__init__(**kwargs)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for free models. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model {model_name} on {self.device}...")
        
        try:
            # Initialize the text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            logger.info(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            # Fallback to a smaller model
            logger.info("Falling back to distilgpt2...")
            self.model_name = "distilgpt2"
            self.pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                device=0 if self.device == "cuda" else -1
            )
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from the prompt."""
        try:
            # Clean up the prompt
            prompt = prompt.strip()
            if len(prompt) > 1000:  # Truncate very long prompts
                prompt = prompt[:1000] + "..."
            
            # Generate response
            result = self.pipeline(
                prompt,
                max_length=min(len(prompt.split()) + self.max_length, 1024),
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                num_return_sequences=1,
                truncation=True
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract only the new generated part
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Clean up the response
            if stop:
                for stop_word in stop:
                    if stop_word in response:
                        response = response.split(stop_word)[0]
            
            return response if response else "I understand your question, but I need more context to provide a helpful answer."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request."
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "device": self.device,
        }


class SimpleLLM(LLM):
    """Simple fallback LLM that works without complex dependencies."""
    
    model_name: str = "simple_fallback"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "simple_fallback"
        logger.info("✅ Using simple fallback LLM")
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a simple response."""
        # Simple keyword-based responses
        prompt_lower = prompt.lower()
        
        if "python" in prompt_lower:
            return "Python is a versatile programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation."
        elif "machine learning" in prompt_lower or "ml" in prompt_lower:
            return "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."
        elif "artificial intelligence" in prompt_lower or "ai" in prompt_lower:
            return "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."
        elif "deep learning" in prompt_lower:
            return "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns."
        else:
            return "I can help answer questions about the loaded documents. Please try asking about topics covered in your documents."
    
    @property
    def _llm_type(self) -> str:
        return "simple"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}


class FreeSentenceTransformerEmbeddings(Embeddings):
    """Free embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a free sentence transformer model.
        
        Popular free models:
        - all-MiniLM-L6-v2: Fast and good quality
        - all-mpnet-base-v2: Higher quality, slower
        - paraphrase-MiniLM-L6-v2: Good for paraphrases
        """
        self.model_name = model_name
        
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")
            
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            logger.info("Using simple embeddings fallback...")
            self.model = None
            self.model_name = "simple_fallback"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if self.model is None:
            # Simple hash-based embeddings as fallback
            return self._simple_embeddings(texts)
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return self._simple_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if self.model is None:
            return self._simple_embeddings([text])[0]
        
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return self._simple_embeddings([text])[0]
    
    def _simple_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create simple hash-based embeddings as fallback."""
        import hashlib
        embeddings = []
        
        for text in texts:
            # Create pseudo-embeddings based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to float values between -1 and 1
            embedding = [(b / 128.0 - 1.0) for b in hash_bytes]
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.extend(embedding[:384 - len(embedding)])
            embedding = embedding[:384]
            embeddings.append(embedding)
        
        return embeddings


def get_free_llm(model_type: str = "conversational") -> LLM:
    """
    Get a free LLM model.
    
    Args:
        model_type: Type of model ("conversational", "general", "small")
    
    Returns:
        Free LLM instance
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available, using simple fallback LLM")
        return SimpleLLM()
    
    models = {
        "conversational": "microsoft/DialoGPT-small",
        "general": "distilgpt2",
        "small": "distilgpt2",
        "medium": "microsoft/DialoGPT-medium"
    }
    
    model_name = models.get(model_type, "microsoft/DialoGPT-small")
    
    try:
        return HuggingFaceLLM(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        logger.info("Using simple fallback LLM...")
        return SimpleLLM()


def get_free_embeddings(model_type: str = "fast") -> Embeddings:
    """
    Get free embeddings model.
    
    Args:
        model_type: Type of embeddings ("fast", "quality", "balanced")
    
    Returns:
        Free embeddings instance
    """
    models = {
        "fast": "all-MiniLM-L6-v2",
        "quality": "all-mpnet-base-v2", 
        "balanced": "paraphrase-MiniLM-L6-v2"
    }
    
    model_name = models.get(model_type, "all-MiniLM-L6-v2")
    return FreeSentenceTransformerEmbeddings(model_name)


def check_model_availability() -> Dict[str, bool]:
    """Check which models are available."""
    status = {
        "transformers": TRANSFORMERS_AVAILABLE,
        "torch": False,
        "sentence_transformers": False
    }
    
    try:
        import torch
        status["torch"] = True
    except ImportError:
        pass
    
    try:
        from sentence_transformers import SentenceTransformer
        status["sentence_transformers"] = True
    except ImportError:
        pass
    
    return status