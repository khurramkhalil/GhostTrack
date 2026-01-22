"""Phi-2 model wrapper with residual stream hooks."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class PhiWithResidualHooks:
    """
    Wrapper around Phi-2 that extracts intermediate activations.

    Extracts:
    1. Residual stream (full block output after attention + MLP)
    2. MLP outputs (separately) - approximation for Phi
    3. Attention outputs (separately) - approximation for Phi

    Note: Phi-2 code structure differs from GPT-2. We hook into:
    - model.layers[i] (residual stream)
    - model.layers[i].mlp (MLP)
    - model.layers[i].self_attn (Attention)
    """

    def __init__(
        self,
        model_name: str = 'microsoft/phi-2',
        device: Optional[str] = None
    ):
        """
        Initialize Phi-2 with hooks.

        Args:
            model_name: HuggingFace model path (e.g., 'microsoft/phi-2')
            device: Device to load model on.
        """
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Phi-2 tokenizer may not have pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model config
        self.config = self.model.config
        # Phi-2 uses 'n_embd' or 'hidden_size' depending on version, checking both
        self.d_model = getattr(self.config, 'n_embd', getattr(self.config, 'hidden_size', 2560))
        # Phi-2 uses 'n_layer' or 'num_hidden_layers'
        self.n_layers = getattr(self.config, 'n_layer', getattr(self.config, 'num_hidden_layers', 32))

        # Storage for hooked activations
        self.cache = {
            'residual_stream': [],
            'mlp_outputs': [],
            'attn_outputs': []
        }

        # Hook handles
        self.hooks = []

        print(f"Model loaded: {self.n_layers} layers, d_model={self.d_model}")

    def _hook_residual_stream(self, layer_idx: int):
        """Create hook for residual stream."""
        def hook(module, input, output):
            # output[0] is hidden states for Phi usually
            # Check tuple or tensor
            if isinstance(output, tuple):
                hidden_states = output[0].detach()
            else:
                hidden_states = output.detach()
            self.cache['residual_stream'].append(hidden_states)
        return hook

    def _hook_mlp_output(self, layer_idx: int):
        """Create hook for MLP output."""
        def hook(module, input, output):
            mlp_output = output.detach()
            self.cache['mlp_outputs'].append(mlp_output)
        return hook

    def _hook_attn_output(self, layer_idx: int):
        """Create hook for attention output."""
        def hook(module, input, output):
            # Attention output might be tuple (context, ...)
            if isinstance(output, tuple):
                attn_output = output[0].detach()
            else:
                attn_output = output.detach()
            self.cache['attn_outputs'].append(attn_output)
        return hook

    def register_hooks(self):
        """Register forward hooks on all layers."""
        print("Registering hooks...")

        # Access layers - typically model.layers or model.model.layers
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError(f"Could not locate layers in {self.model_name}")

        for layer_idx, block in enumerate(layers):
            # Hook full block (residual stream)
            handle = block.register_forward_hook(
                self._hook_residual_stream(layer_idx)
            )
            self.hooks.append(handle)

            # Hook MLP
            if hasattr(block, 'mlp'):
                handle = block.mlp.register_forward_hook(
                    self._hook_mlp_output(layer_idx)
                )
                self.hooks.append(handle)

            # Hook attention
            if hasattr(block, 'self_attn'):
                handle = block.self_attn.register_forward_hook(
                    self._hook_attn_output(layer_idx)
                )
                self.hooks.append(handle)

        print(f"Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("Removed all hooks")

    def clear_cache(self):
        """Clear cached activations."""
        self.cache = {
            'residual_stream': [],
            'mlp_outputs': [],
            'attn_outputs': []
        }

    @torch.no_grad()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """Forward pass with activation caching."""
        self.clear_cache()

        if len(self.hooks) == 0:
            self.register_hooks()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        result = {
            'logits': outputs.logits,
            'residual_stream': self.cache['residual_stream'],
            'mlp_outputs': self.cache['mlp_outputs'],
            'attn_outputs': self.cache['attn_outputs']
        }

        return result

    def encode_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode text to token IDs."""
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )

        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    def process_text(self, text: str, max_length: int = 512) -> Dict[str, any]:
        """Encode text and run forward pass with caching."""
        encoded = self.encode_text(text, max_length)
        result = self.forward_with_cache(
            encoded['input_ids'],
            encoded['attention_mask']
        )
        result['attention_mask'] = encoded['attention_mask']
        return result

    def get_activation_shape(self) -> Tuple[int, int]:
        """Get expected shape of activations."""
        return self.n_layers, self.d_model

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'hooks'):
            self.remove_hooks()
