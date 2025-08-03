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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLM):
    """Free Hugging Face LLM implementation."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 512,
        temperature: float = 0.7,
        device: str = "auto"
    ):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            device: Device to run the model on
        """
        super().__init__()
        
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
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"


class FreeConversationalLLM(LLM):
    """Optimized free LLM for conversational tasks."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading conversational model {model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate conversational response."""
        try:
            # Format prompt for conversation
            formatted_prompt = f"Context: {prompt}\n\nResponse:"
            
            # Tokenize
            inputs = self.tokenizer.encode(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part
            if "Response:" in response:
                response = response.split("Response:")[-1].strip()
            
            return response if response else "I understand your question, but need more details to help you better."
            
        except Exception as e:
            logger.error(f"Error in conversation generation: {e}")
            return "I apologize for the technical difficulty. Could you please rephrase your question?"
    
    @property
    def _llm_type(self) -> str:
        return "free_conversational"


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
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 384


def get_free_llm(model_type: str = "conversational") -> LLM:
    """
    Get a free LLM model.
    
    Args:
        model_type: Type of model ("conversational", "general", "small")
    
    Returns:
        Free LLM instance
    """
    models = {
        "conversational": "microsoft/DialoGPT-small",
        "general": "distilgpt2",
        "small": "distilgpt2",
        "medium": "microsoft/DialoGPT-medium"
    }
    
    model_name = models.get(model_type, "microsoft/DialoGPT-small")
    
    try:
        if model_type == "conversational":
            return FreeConversationalLLM(model_name)
        else:
            return HuggingFaceLLM(model_name)
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        # Ultimate fallback
        return HuggingFaceLLM("distilgpt2")


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