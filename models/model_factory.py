"""
Model factory for multi-model support.

Provides a unified interface for loading different model architectures
with residual stream hooks for GhostTrack analysis.

Supported Models:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Future: Phi, Qwen, Llama families

Usage:
    from models import get_model_wrapper
    
    model = get_model_wrapper('gpt2-medium', device='cuda')
    # or
    model = get_model_wrapper('microsoft/phi-2', device='cuda')
"""

from typing import Optional, Dict, Type
import warnings

from .model_wrapper import GPT2WithResidualHooks
from .phi_wrapper import PhiWithResidualHooks


# Registry mapping model name prefixes to wrapper classes
# Add new model families here as they are implemented
MODEL_REGISTRY: Dict[str, Type] = {
    # GPT-2 family
    'gpt2': GPT2WithResidualHooks,
    
    # Phi family
    'phi': PhiWithResidualHooks,
    # 'qwen': QwenWithResidualHooks,  
    # 'llama': LlamaWithResidualHooks,
    # 'mistral': MistralWithResidualHooks,
}

# Model family aliases for HuggingFace model paths
HF_MODEL_FAMILIES = {
    'microsoft/phi': 'phi',
    'qwen/qwen': 'qwen',
    'meta-llama': 'llama',
    'mistralai': 'mistral',
}


def get_model_wrapper(
    model_name: str, 
    device: Optional[str] = None,
    **kwargs
):
    """
    Get appropriate model wrapper for the given model.
    
    This factory function selects the correct wrapper class based on the
    model name and instantiates it with the provided configuration.
    
    Args:
        model_name: Model identifier. Can be:
            - Short name: 'gpt2', 'gpt2-medium', 'phi-2'
            - HuggingFace path: 'microsoft/phi-2', 'Qwen/Qwen2-1.5B'
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        **kwargs: Additional arguments passed to the wrapper constructor
        
    Returns:
        Model wrapper instance with residual hooks configured
        
    Raises:
        ValueError: If model family is not yet supported
        
    Examples:
        >>> model = get_model_wrapper('gpt2-medium', device='cuda')
        >>> model = get_model_wrapper('microsoft/phi-2', device='cuda')
    """
    # Normalize model name
    normalized = model_name.lower()
    
    # Check for HuggingFace path format
    for hf_prefix, family in HF_MODEL_FAMILIES.items():
        if normalized.startswith(hf_prefix.lower()):
            wrapper_cls = MODEL_REGISTRY.get(family)
            if wrapper_cls is None:
                raise ValueError(
                    f"Model family '{family}' (from '{model_name}') is not yet supported. "
                    f"Supported families: {list(MODEL_REGISTRY.keys())}"
                )
            return wrapper_cls(model_name=model_name, device=device, **kwargs)
    
    # Check for direct model name matches
    for prefix, wrapper_cls in MODEL_REGISTRY.items():
        if normalized.startswith(prefix):
            return wrapper_cls(model_name=model_name, device=device, **kwargs)
    
    # Unknown model - warn and try GPT-2 style wrapper
    # This may work for decoder-only transformers with similar architecture
    warnings.warn(
        f"Unknown model '{model_name}'. Attempting GPT-2 style wrapper. "
        f"This may fail for incompatible architectures. "
        f"Supported prefixes: {list(MODEL_REGISTRY.keys())}",
        UserWarning
    )
    return GPT2WithResidualHooks(model_name=model_name, device=device, **kwargs)


def get_supported_models() -> Dict[str, str]:
    """
    Get dictionary of supported model families and their status.
    
    Returns:
        Dict mapping model family to implementation status
    """
    status = {}
    for family in MODEL_REGISTRY:
        status[family] = 'implemented'
    
    # Add known future models
    future_models = ['phi', 'qwen', 'llama', 'mistral']
    for model in future_models:
        if model not in status:
            status[model] = 'planned'
    
    return status


def register_model_wrapper(prefix: str, wrapper_cls: Type) -> None:
    """
    Register a new model wrapper class.
    
    Use this to add support for new model families at runtime.
    
    Args:
        prefix: Model name prefix to match (e.g., 'phi', 'llama')
        wrapper_cls: Wrapper class to use for this model family
        
    Example:
        >>> from models.phi_wrapper import PhiWithResidualHooks
        >>> register_model_wrapper('phi', PhiWithResidualHooks)
    """
    MODEL_REGISTRY[prefix.lower()] = wrapper_cls
