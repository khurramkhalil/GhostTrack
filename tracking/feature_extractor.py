"""
Layerwise feature extraction for hypothesis tracking.

Extracts SAE features from model activations for tracking.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class LayerwiseFeatureExtractor:
    """
    Extract SAE features for each layer to use in hypothesis tracking.
    """

    def __init__(
        self,
        model_wrapper,
        saes: List,
        device: str = 'cuda'
    ):
        """
        Initialize feature extractor.

        Args:
            model_wrapper: GPT2WithResidualHooks instance.
            saes: List of trained JumpReLUSAE models (one per layer).
            device: Device to use.
        """
        self.model = model_wrapper
        self.saes = [sae.to(device).eval() for sae in saes]
        self.device = device
        self.n_layers = len(saes)

        assert self.n_layers == self.model.n_layers, \
            f"Number of SAEs ({self.n_layers}) must match model layers ({self.model.n_layers})"

    def extract_features(self, text: str, max_length: int = 512) -> List[Dict]:
        """
        Extract SAE features for all layers from input text.

        Args:
            text: Input text string.
            max_length: Maximum sequence length.

        Returns:
            List of dicts with layer features:
            [{
                'layer': int,
                'features': tensor [seq_len, d_hidden],
                'activations': tensor [seq_len, d_hidden],
                'error': tensor [seq_len, d_model],
                'hidden_state': tensor [seq_len, d_model]
            }, ...]
        """
        # Get model activations
        outputs = self.model.process_text(text, max_length=max_length)

        layer_features = []

        with torch.no_grad():
            for layer_idx in range(self.n_layers):
                # Get hidden states for this layer
                hidden = outputs['residual_stream'][layer_idx]  # [1, seq_len, d_model]

                # Pass through SAE
                sae_output = self.saes[layer_idx].forward(hidden)

                layer_features.append({
                    'layer': layer_idx,
                    'features': sae_output['features'][0],  # [seq_len, d_hidden]
                    'activations': sae_output['features'][0],  # Same, for clarity
                    'error': sae_output['error'][0],  # [seq_len, d_model]
                    'hidden_state': hidden[0],  # [seq_len, d_model]
                    'sparsity': sae_output['sparsity'].item()
                })

        return layer_features

    def get_top_k_features(
        self,
        layer_features: Dict,
        k: int = 50,
        token_pos: Optional[int] = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Get top-k activated features from a layer.

        Args:
            layer_features: Dict from extract_features for one layer.
            k: Number of top features to return.
            token_pos: Specific token position (if None, uses max over all positions).

        Returns:
            List of (feature_id, activation, embedding) tuples.
        """
        features = layer_features['activations']  # [seq_len, d_hidden]

        if token_pos is not None:
            # Use specific token position
            feature_vector = features[token_pos].cpu().numpy()
        else:
            # Use max activation across all positions for each feature
            feature_vector = features.max(dim=0)[0].cpu().numpy()

        # Get top-k indices
        top_k_indices = np.argsort(feature_vector)[::-1][:k]

        # Get feature embeddings from SAE decoder
        sae = self.saes[layer_features['layer']]
        decoder_weights = sae.W_dec.weight.data.cpu().numpy()  # [d_model, d_hidden]

        result = []
        for feat_id in top_k_indices:
            activation = feature_vector[feat_id]
            # Embedding is the decoder column for this feature
            embedding = decoder_weights[:, feat_id]  # [d_model]
            result.append((int(feat_id), float(activation), embedding))

        return result

    def extract_at_position(
        self,
        text: str,
        position: int,
        k: int = 50
    ) -> List[Dict]:
        """
        Extract top-k features at a specific token position for all layers.

        Args:
            text: Input text.
            position: Token position.
            k: Number of top features per layer.

        Returns:
            List of dicts per layer with top-k features.
        """
        layer_features = self.extract_features(text)

        results = []
        for layer_data in layer_features:
            # Get features at this position
            features_at_pos = layer_data['activations'][position].cpu().numpy()

            # Get top-k
            top_k_indices = np.argsort(features_at_pos)[::-1][:k]

            # Get decoder embeddings
            sae = self.saes[layer_data['layer']]
            decoder_weights = sae.W_dec.weight.data.cpu().numpy()

            top_features = []
            for feat_id in top_k_indices:
                activation = features_at_pos[feat_id]
                if activation > 0:  # Only include active features
                    embedding = decoder_weights[:, feat_id]
                    top_features.append((int(feat_id), float(activation), embedding))

            results.append({
                'layer': layer_data['layer'],
                'position': position,
                'top_features': top_features,
                'num_active': len(top_features)
            })

        return results

    @classmethod
    def load_from_checkpoints(
        cls,
        model_wrapper,
        checkpoint_dir: str,
        device: str = 'cuda'
    ) -> 'LayerwiseFeatureExtractor':
        """
        Load SAEs from checkpoints and create extractor.

        Args:
            model_wrapper: GPT2WithResidualHooks instance.
            checkpoint_dir: Directory containing SAE checkpoints.
            device: Device to use.

        Returns:
            LayerwiseFeatureExtractor instance.
        """
        from models import JumpReLUSAE

        checkpoint_dir = Path(checkpoint_dir)
        saes = []

        for layer_idx in range(model_wrapper.n_layers):
            checkpoint_path = checkpoint_dir / f'sae_layer_{layer_idx}_best.pt'

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAE checkpoint not found: {checkpoint_path}\n"
                    f"Please train SAEs first (Phase 2)"
                )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Get dimensions from state dict
            state_dict = checkpoint['model_state_dict']
            d_model = state_dict['W_enc.weight'].shape[1]
            d_hidden = state_dict['W_enc.weight'].shape[0]

            # Create SAE and load weights
            sae = JumpReLUSAE(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(state_dict)
            sae.eval()

            saes.append(sae)

            print(f"Loaded SAE for layer {layer_idx}: "
                  f"d_model={d_model}, d_hidden={d_hidden}, "
                  f"loss={checkpoint['best_loss']:.6f}")

        return cls(model_wrapper, saes, device)
