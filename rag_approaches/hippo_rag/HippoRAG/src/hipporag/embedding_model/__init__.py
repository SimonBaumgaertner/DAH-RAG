from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _create_generic_embedding_wrapper(embedding_model_name: str):
    """
    Creates a generic embedding model wrapper for unknown embedding model names.
    This is used as a fallback when the embedding model name is not recognized.
    """
    class GenericEmbeddingWrapper(BaseEmbeddingModel):
        def __init__(self, global_config=None, embedding_model_name: str = ""):
            super().__init__(global_config=global_config)
            self.embedding_model_name = embedding_model_name
            # Set a default embedding dimension - this should be overridden by the actual encoder
            self.embedding_dim = 1024
            self.embedding_config = EmbeddingConfig.from_dict({
                "embedding_model_name": embedding_model_name,
                "embedding_dim": self.embedding_dim,
            })
            logger.warning(f"Using generic embedding wrapper for: {embedding_model_name}")
        
        def batch_encode(self, texts, **kwargs):
            # This should never be called directly - the patching mechanism should override this
            raise NotImplementedError(
                f"Generic embedding wrapper for {self.embedding_model_name} was not properly patched. "
                "This usually means the patching mechanism failed to replace this method."
            )
    
    return GenericEmbeddingWrapper


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    else:
        # Fallback for unknown embedding models - check if there's a custom embedding class available
        # This allows for patching mechanisms to work with custom embedding models
        import sys
        import inspect
        
        # Look for a custom embedding class in the calling frame
        frame = inspect.currentframe()
        try:
            # Check if there's a patched embedding getter in the module
            calling_module = frame.f_back.f_globals.get('__name__', '')
            if calling_module and hasattr(sys.modules.get(calling_module, None), '_get_embedding_model_class'):
                # There's a patched version available, use it
                patched_getter = getattr(sys.modules[calling_module], '_get_embedding_model_class')
                if callable(patched_getter):
                    return patched_getter(embedding_model_name)
            
            # If no patched version, create a generic wrapper that can handle any embedding model name
            logger.warning(f"Unknown embedding model name: {embedding_model_name}. Using generic fallback.")
            return _create_generic_embedding_wrapper(embedding_model_name)
        finally:
            del frame