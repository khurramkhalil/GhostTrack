"""GPT-2 model wrapper with residual stream hooks."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2WithResidualHooks:
    """
    Wrapper around GPT-2 that extracts intermediate activations.

    Extracts:
    1. Residual stream (full block output after attention + MLP)
    2. MLP outputs (separately)
    3. Attention outputs (separately)

    This allows us to analyze how information flows through the network
    and track semantic hypotheses across layers.
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        device: Optional[str] = None
    ):
        """
        Initialize GPT-2 with hooks.

        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
            device: Device to load model on. If None, uses cuda if available.
        """
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading {model_name} on {self.device}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model config
        self.config = self.model.config
        self.n_layers = self.config.n_layer
        self.d_model = self.config.n_embd

        # Storage for hooked activations
        self.cache = {
            'residual_stream': [],
            'mlp_outputs': [],
            'attn_outputs': []
        }

        # Hook handles (for removal)
        self.hooks = []

        print(f"Model loaded: {self.n_layers} layers, d_model={self.d_model}")

    def _hook_residual_stream(self, layer_idx: int):
        """Create hook for residual stream (full block output)."""
        def hook(module, input, output):
            # output[0] is the hidden states after full transformer block
            hidden_states = output[0].detach()
            self.cache['residual_stream'].append(hidden_states)
        return hook

    def _hook_mlp_output(self, layer_idx: int):
        """Create hook for MLP output."""
        def hook(module, input, output):
            # output is the MLP output
            mlp_output = output.detach()
            self.cache['mlp_outputs'].append(mlp_output)
        return hook

    def _hook_attn_output(self, layer_idx: int):
        """Create hook for attention output."""
        def hook(module, input, output):
            # output[0] is the attention output
            attn_output = output[0].detach()
            self.cache['attn_outputs'].append(attn_output)
        return hook

    def register_hooks(self):
        """
        Register forward hooks on all layers.

        Hooks are registered on:
        - transformer.h[i] (full residual stream)
        - transformer.h[i].mlp (MLP only)
        - transformer.h[i].attn (attention only)
        """
        print("Registering hooks...")

        for layer_idx in range(self.n_layers):
            # Hook full block (residual stream)
            block = self.model.transformer.h[layer_idx]
            handle = block.register_forward_hook(
                self._hook_residual_stream(layer_idx)
            )
            self.hooks.append(handle)

            # Hook MLP
            mlp = block.mlp
            handle = mlp.register_forward_hook(
                self._hook_mlp_output(layer_idx)
            )
            self.hooks.append(handle)

            # Hook attention
            attn = block.attn
            handle = attn.register_forward_hook(
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
        """
        Forward pass with activation caching.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dictionary with:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - residual_stream: List of tensors [batch_size, seq_len, d_model]
            - mlp_outputs: List of tensors [batch_size, seq_len, d_model]
            - attn_outputs: List of tensors [batch_size, seq_len, d_model]
        """
        # Clear previous cache
        self.clear_cache()

        # Ensure hooks are registered
        if len(self.hooks) == 0:
            self.register_hooks()

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

        # Collect results
        result = {
            'logits': outputs.logits,
            'residual_stream': self.cache['residual_stream'],
            'mlp_outputs': self.cache['mlp_outputs'],
            'attn_outputs': self.cache['attn_outputs']
        }

        # Validate cache sizes
        assert len(result['residual_stream']) == self.n_layers, \
            f"Expected {self.n_layers} residual stream activations, got {len(result['residual_stream'])}"
        assert len(result['mlp_outputs']) == self.n_layers, \
            f"Expected {self.n_layers} MLP activations, got {len(result['mlp_outputs'])}"
        assert len(result['attn_outputs']) == self.n_layers, \
            f"Expected {self.n_layers} attention activations, got {len(result['attn_outputs'])}"

        return result

    def encode_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
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
        """
        Encode text and run forward pass with caching.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with logits and all cached activations.
        """
        encoded = self.encode_text(text, max_length)
        return self.forward_with_cache(
            encoded['input_ids'],
            encoded['attention_mask']
        )

    def get_activation_shape(self) -> Tuple[int, int]:
        """
        Get expected shape of activations.

        Returns:
            Tuple of (n_layers, d_model).
        """
        return self.n_layers, self.d_model

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'hooks'):
            self.remove_hooks()
