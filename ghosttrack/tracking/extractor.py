"""
Feature extractor — runs model + SAEs to produce per-layer sparse features.

:class:`FeatureExtractor` bridges the model wrappers and the hypothesis
tracker by extracting SAE features from every layer of a forward pass.

Named tuple :class:`LayerFeatures` is the standard data structure passed
to :class:`~ghosttrack.tracking.tracker.HypothesisTracker`.

Usage::

    from ghosttrack.tracking import FeatureExtractor
    extractor = FeatureExtractor.from_checkpoints(model, "/data/sae_checkpoints")
    layer_features = extractor.extract("Paris is the capital of France.")
    # layer_features[i] → LayerFeatures(layer=i, activations=..., feature_ids=..., ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import torch

from ghosttrack.sae.model import JumpReLUSAE


class LayerFeatures(NamedTuple):
    """Per-layer output of :class:`FeatureExtractor`."""

    layer: int
    """Layer index."""

    activations: torch.Tensor
    """SAE feature activations ``[seq_len, d_hidden]`` (sparse)."""

    feature_ids: np.ndarray
    """Indices of the top-k active features ``[k]``."""

    error: torch.Tensor
    """Reconstruction error ``[seq_len, d_model]``."""

    hidden_state: torch.Tensor
    """Raw residual-stream activations ``[seq_len, d_model]``."""

    sparsity: float
    """Fraction of active features in this layer."""


class FeatureExtractor:
    """
    Extract SAE features for every layer of a model forward pass.

    Args:
        model: A :class:`~ghosttrack.models.base.BaseModelWrapper` instance.
        saes: List of trained :class:`~ghosttrack.sae.model.JumpReLUSAE`
            models — one per transformer layer.
        device: Computation device.  Inherited from *model* when ``None``.
    """

    def __init__(self, model, saes: List[JumpReLUSAE], device: Optional[str] = None):
        self.model = model
        self.device = device or model.device
        self.saes = [sae.to(self.device).eval() for sae in saes]

        if len(self.saes) != model.n_layers:
            raise ValueError(
                f"Number of SAEs ({len(self.saes)}) must equal "
                f"model.n_layers ({model.n_layers})"
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, text: str, max_length: int = 512) -> List[LayerFeatures]:
        """
        Run a forward pass on *text* and extract SAE features per layer.

        Args:
            text: Input string.
            max_length: Tokeniser max length.

        Returns:
            List of :class:`LayerFeatures` — one per transformer layer.
        """
        outputs = self.model.process_text(text, max_length=max_length)
        return self._run_saes(outputs)

    def extract_batch(
        self, texts: List[str], max_length: int = 512
    ) -> List[List[LayerFeatures]]:
        """
        Extract features for a batch of texts.

        Returns:
            ``[text_idx][layer_idx]`` nested list of :class:`LayerFeatures`.
            Each entry's tensors contain only the non-padding tokens for
            that text.
        """
        encoded = self.model.encode_text(texts[0], max_length)  # warmup tokeniser
        # Use process_text per item to keep the single-text API simple
        return [self.extract(text, max_length) for text in texts]

    def get_top_k_features(
        self,
        layer_feat: LayerFeatures,
        k: int = 50,
        token_pos: Optional[int] = None,
    ) -> List:
        """
        Return the *k* most activated ``(feature_id, activation, decoder_embedding)``
        tuples for one layer.

        Args:
            layer_feat: :class:`LayerFeatures` for a single layer.
            k: Number of features to return.
            token_pos: If ``None``, pool over the whole sequence (max).

        Returns:
            ``[(int, float, np.ndarray), ...]`` sorted by activation descending.
        """
        acts = layer_feat.activations  # [seq_len, d_hidden]
        if token_pos is not None:
            vec = acts[token_pos].cpu().numpy()
        else:
            vec = acts.max(dim=0)[0].cpu().numpy()

        sae = self.saes[layer_feat.layer]
        decoder = sae.W_dec.weight.data.cpu().numpy()  # [d_model, d_hidden]

        top_k_indices = np.argsort(vec)[::-1][:k]

        result = []
        for feat_id in top_k_indices:
            activation = float(vec[feat_id])
            # .copy() makes an independent array so the 20 MB `decoder`
            # can be freed when this function returns (column slices are
            # views in numpy and would otherwise keep the whole base alive).
            embedding = decoder[:, feat_id].copy()
            result.append((int(feat_id), activation, embedding))

        return result

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoints(
        cls,
        model,
        checkpoint_dir: str,
        device: Optional[str] = None,
    ) -> "FeatureExtractor":
        """
        Load SAE checkpoints and build a :class:`FeatureExtractor`.

        Args:
            model: Initialised model wrapper.
            checkpoint_dir: Directory containing ``sae_layer_*_best.pt`` files.
            device: Target device.

        Returns:
            Ready-to-use :class:`FeatureExtractor`.
        """
        dev = device or model.device
        ckpt_dir = Path(checkpoint_dir)
        saes: List[JumpReLUSAE] = []

        for layer_idx in range(model.n_layers):
            ckpt_path = ckpt_dir / f"sae_layer_{layer_idx}_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"SAE checkpoint not found: {ckpt_path}\n"
                    "Train SAEs first (Phase 2)."
                )
            ckpt = torch.load(ckpt_path, map_location=dev)
            state = ckpt["model_state_dict"]
            d_model = state["W_enc.weight"].shape[1]
            d_hidden = state["W_enc.weight"].shape[0]

            sae = JumpReLUSAE(d_model=d_model, d_hidden=d_hidden)
            sae.load_state_dict(state)
            sae.eval()
            saes.append(sae)

        return cls(model, saes, device=dev)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_saes(self, model_outputs: dict) -> List[LayerFeatures]:
        """Run all SAEs on cached hidden states."""
        results: List[LayerFeatures] = []
        with torch.no_grad():
            for i, sae in enumerate(self.saes):
                hidden = model_outputs["residual_stream"][i]  # [1, seq, d_model]
                sae_out = sae(hidden.to(self.device, dtype=torch.float32))

                acts = sae_out.features[0]         # [seq, d_hidden]
                err = sae_out.error[0]             # [seq, d_model]
                h = hidden[0]                       # [seq, d_model]

                active_ids = (acts.max(dim=0)[0] > 0).nonzero(as_tuple=True)[0].cpu().numpy()

                results.append(LayerFeatures(
                    layer=i,
                    activations=acts,
                    feature_ids=active_ids,
                    error=err,
                    hidden_state=h,
                    sparsity=float(sae_out.sparsity),
                ))
        return results
