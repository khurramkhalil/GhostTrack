"""JumpReLU Sparse Autoencoder implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Based on Rajamanoharan et al., 2024.
    Provides better reconstruction-sparsity tradeoff than standard ReLU SAE.

    JumpReLU activation: f(x) = x if x > threshold, else 0
    The threshold is learned during training.
    """

    def __init__(
        self,
        d_model: int = 768,
        d_hidden: int = 4096,
        threshold: float = 0.1,
        lambda_sparse: float = 0.01
    ):
        """
        Initialize JumpReLU SAE.

        Args:
            d_model: Input/output dimension (e.g., 768 for GPT-2).
            d_hidden: Hidden layer dimension (typically 4-8x d_model).
            threshold: Initial threshold for JumpReLU activation.
            lambda_sparse: Weight for L1 sparsity penalty.
        """
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.lambda_sparse = lambda_sparse

        # Encoder: projects from d_model to d_hidden
        self.W_enc = nn.Linear(d_model, d_hidden, bias=True)

        # Decoder: projects from d_hidden back to d_model
        self.W_dec = nn.Linear(d_hidden, d_model, bias=True)

        # Learned threshold for JumpReLU
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # Initialize decoder with normalized columns
        self.normalize_decoder()

    def normalize_decoder(self):
        """
        Normalize decoder weight columns to unit norm.

        This is a standard technique for SAEs that helps with
        interpretability and training stability.

        W_dec.weight is [d_model, d_hidden]. Each column (along dim 1)
        represents one feature's decoder vector. We normalize columns so each
        feature has unit norm.
        """
        with torch.no_grad():
            # W_dec.weight is [d_model, d_hidden]
            # Normalize each column (feature decoder) to unit norm
            norms = torch.norm(self.W_dec.weight.data, dim=0, keepdim=True)
            self.W_dec.weight.data = self.W_dec.weight.data / (norms + 1e-8)

    def jumprelu(self, x: torch.Tensor) -> torch.Tensor:
        """
        JumpReLU activation function.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor: x if x > threshold, else 0.
        """
        # Create mask for values above threshold
        mask = (x > self.threshold).float()
        return x * mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature space.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].

        Returns:
            Encoded features [batch_size, seq_len, d_hidden].
        """
        # Linear projection
        pre_activation = self.W_enc(x)

        # JumpReLU activation
        features = self.jumprelu(pre_activation)

        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to original space.

        Args:
            features: Encoded features [batch_size, seq_len, d_hidden].

        Returns:
            Reconstructed input [batch_size, seq_len, d_model].
        """
        return self.W_dec(features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SAE.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].

        Returns:
            Dictionary containing:
            - reconstruction: Reconstructed input [batch_size, seq_len, d_model]
            - features: Sparse features [batch_size, seq_len, d_hidden]
            - error: Reconstruction error (x - reconstruction)
            - sparsity: Fraction of active features
        """
        # Encode
        features = self.encode(x)

        # Decode
        reconstruction = self.decode(features)

        # Compute error (important: hallucinations may hide here)
        error = x - reconstruction

        # Compute sparsity (fraction of active features)
        sparsity = (features > 0).float().mean()

        return {
            'reconstruction': reconstruction,
            'features': features,
            'error': error,
            'sparsity': sparsity
        }

    def loss(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute total loss (reconstruction + sparsity).

        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            return_components: If True, return dict with loss components.

        Returns:
            Total loss (scalar) or dict with loss components.
        """
        # Forward pass
        output = self.forward(x)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(output['reconstruction'], x)

        # Sparsity loss (L1 penalty on activations)
        sparsity_loss = output['features'].abs().mean()

        # Total loss
        total_loss = recon_loss + self.lambda_sparse * sparsity_loss

        if return_components:
            return {
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'sparsity_loss': sparsity_loss,
                'sparsity': output['sparsity'],
                'threshold': self.threshold.item()
            }
        else:
            return total_loss

    def get_feature_norms(self) -> torch.Tensor:
        """
        Get L2 norms of decoder weight columns (features).

        Returns:
            Tensor of shape [d_hidden] with norm of each feature decoder.
        """
        # W_dec.weight is [d_model, d_hidden], compute norm over d_model dimension (dim=0)
        return torch.norm(self.W_dec.weight.data, dim=0)

    def get_active_features(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get mask of active features for input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            threshold: Activation threshold. If None, uses learned threshold.

        Returns:
            Boolean mask [batch_size, seq_len, d_hidden].
        """
        features = self.encode(x)

        if threshold is None:
            threshold = self.threshold.item()

        return features > threshold

    def count_active_features(self, x: torch.Tensor) -> float:
        """
        Count average number of active features per token.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].

        Returns:
            Average number of active features.
        """
        active_mask = self.get_active_features(x)
        return active_mask.float().sum(dim=-1).mean().item()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize decoder during training (call periodically)."""
        self.normalize_decoder()

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
