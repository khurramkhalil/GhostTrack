"""
Feature interpretation for trained SAEs.

Analyzes what each SAE feature represents by finding
top-activating examples and extracting patterns.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm
import json


class FeatureInterpreter:
    """
    Interpret SAE features by analyzing activations.
    """

    def __init__(
        self,
        sae,
        model_wrapper,
        layer_idx: int,
        device: str = 'cuda'
    ):
        """
        Initialize feature interpreter.

        Args:
            sae: Trained JumpReLUSAE model.
            model_wrapper: GPT2WithResidualHooks for getting activations.
            layer_idx: Layer index this SAE was trained on.
            device: Device to use.
        """
        self.sae = sae.to(device)
        self.sae.eval()
        self.model = model_wrapper
        self.layer_idx = layer_idx
        self.device = device

    def get_feature_activations(
        self,
        text: str,
        feature_id: int,
        max_length: int = 512
    ) -> np.ndarray:
        """
        Get activations for a specific feature across all tokens in text.

        Args:
            text: Input text.
            feature_id: Feature ID to get activations for.
            max_length: Max sequence length.

        Returns:
            Array of activations (one per token).
        """
        with torch.no_grad():
            # Get model activations
            outputs = self.model.process_text(text, max_length=max_length)
            layer_states = outputs['residual_stream'][self.layer_idx]

            # Pass through SAE
            sae_output = self.sae.forward(layer_states)
            features = sae_output['features']  # [1, seq_len, d_hidden]

            # Get activations for this feature
            feature_acts = features[0, :, feature_id].cpu().numpy()

            return feature_acts

    def find_top_activating_examples(
        self,
        texts: List[str],
        feature_id: int,
        top_k: int = 20,
        max_length: int = 512
    ) -> List[Dict]:
        """
        Find top-k examples that activate a specific feature.

        Args:
            texts: List of text examples to search.
            feature_id: Feature ID to analyze.
            top_k: Number of top examples to return.
            max_length: Max sequence length.

        Returns:
            List of dicts with 'text', 'max_activation', 'token_idx', 'activations'.
        """
        top_activations = []  # (text, position, activation, all_activations)

        with torch.no_grad():
            for text in texts:
                # Get activations for this text
                activations = self.get_feature_activations(text, feature_id, max_length)

                # Find max activation in this text
                max_idx = np.argmax(activations)
                max_act = activations[max_idx]

                if max_act > 0:  # Only if feature is active
                    top_activations.append({
                        'text': text,
                        'max_activation': float(max_act),
                        'token_idx': int(max_idx),
                        'activations': activations
                    })

        # Sort by activation and take top-k
        top_activations.sort(key=lambda x: x['max_activation'], reverse=True)
        return top_activations[:top_k]

    def extract_common_tokens(
        self,
        examples: List[Tuple[str, int, float]],
        context_window: int = 5
    ) -> Dict[str, int]:
        """
        Extract common tokens around activation positions.

        Args:
            examples: List of (text, position, activation) tuples.
            context_window: Number of tokens before/after to include.

        Returns:
            Dictionary of token -> count.
        """
        token_counts = Counter()

        for text, pos, _ in examples:
            # Tokenize
            tokens = self.model.tokenizer.encode(text)

            # Extract context window
            start = max(0, pos - context_window)
            end = min(len(tokens), pos + context_window + 1)

            context_tokens = tokens[start:end]

            # Decode to strings
            context_strings = [
                self.model.tokenizer.decode([t])
                for t in context_tokens
            ]

            # Count
            token_counts.update(context_strings)

        return dict(token_counts)

    def get_common_tokens(
        self,
        texts: List[str],
        feature_id: int,
        top_k: int = 10,
        context_window: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Get most common tokens where feature activates.

        Args:
            texts: List of texts to analyze.
            feature_id: Feature ID.
            top_k: Number of top tokens to return.
            context_window: Context window around activation.

        Returns:
            List of (token, count) tuples.
        """
        # Find top activating examples
        examples = self.find_top_activating_examples(texts, feature_id, top_k=20)

        # Convert to old format for extract_common_tokens
        examples_old_format = [
            (ex['text'], ex['token_idx'], ex['max_activation'])
            for ex in examples
        ]

        # Extract common tokens
        token_counts = self.extract_common_tokens(examples_old_format, context_window)

        # Sort by count
        sorted_tokens = sorted(
            token_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_tokens[:top_k]

    def identify_dead_features(
        self,
        texts: List[str],
        threshold: float = 1e-6,
        max_length: int = 512
    ) -> List[int]:
        """
        Identify features that never activate above threshold.

        Args:
            texts: List of texts to check.
            threshold: Minimum activation to be considered alive.
            max_length: Max sequence length.

        Returns:
            List of dead feature IDs.
        """
        d_hidden = self.sae.d_hidden
        max_activations = np.zeros(d_hidden)

        with torch.no_grad():
            for text in texts:
                # Get model activations
                outputs = self.model.process_text(text, max_length=max_length)
                layer_states = outputs['residual_stream'][self.layer_idx]

                # Pass through SAE
                sae_output = self.sae.forward(layer_states)
                features = sae_output['features']  # [1, seq_len, d_hidden]

                # Get max activation for each feature
                batch_max = features[0].max(dim=0)[0].cpu().numpy()
                max_activations = np.maximum(max_activations, batch_max)

        # Find dead features
        dead_features = [
            i for i in range(d_hidden)
            if max_activations[i] < threshold
        ]

        return dead_features

    def get_feature_statistics(
        self,
        texts: List[str],
        feature_id: int,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Compute statistics for a feature across texts.

        Args:
            texts: List of texts.
            feature_id: Feature ID.
            max_length: Max sequence length.

        Returns:
            Dict with statistics.
        """
        all_activations = []
        num_active = 0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                activations = self.get_feature_activations(text, feature_id, max_length)
                all_activations.extend(activations)
                total_tokens += len(activations)

                # Count tokens where feature is active
                num_active += np.sum(activations > 1e-6)

        all_activations = np.array(all_activations)

        return {
            'mean_activation': float(np.mean(all_activations)),
            'max_activation': float(np.max(all_activations)),
            'activation_frequency': num_active / total_tokens if total_tokens > 0 else 0.0,
            'num_samples': len(texts)
        }

    def interpret_feature(
        self,
        feature_id: int,
        texts: List[str],
        k: int = 20
    ) -> Dict:
        """
        Interpret a single feature.

        Args:
            feature_id: Feature ID to interpret.
            texts: Text corpus for analysis.
            k: Number of examples to analyze.

        Returns:
            Dictionary with interpretation results.
        """
        # Find top examples
        top_examples = self.find_top_activating_examples(
            texts, feature_id, top_k=k
        )

        if not top_examples:
            return {
                'feature_id': feature_id,
                'active': False,
                'top_activation': 0.0
            }

        # Extract common tokens - convert to old format
        examples_old_format = [
            (ex['text'], ex['token_idx'], ex['max_activation'])
            for ex in top_examples
        ]
        common_tokens = self.extract_common_tokens(examples_old_format)

        # Sort by frequency
        sorted_tokens = sorted(
            common_tokens.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        return {
            'feature_id': feature_id,
            'active': True,
            'num_activations': len(top_examples),
            'top_activation': top_examples[0]['max_activation'],
            'mean_activation': np.mean([ex['max_activation'] for ex in top_examples]),
            'common_tokens': sorted_tokens,
            'top_examples': [
                {
                    'text': ex['text'][:200],  # Truncate for storage
                    'position': ex['token_idx'],
                    'activation': ex['max_activation']
                }
                for ex in top_examples[:5]
            ]
        }

    def interpret_all_features(
        self,
        texts: List[str],
        k: int = 20,
        save_path: Optional[str] = None
    ) -> Dict[int, Dict]:
        """
        Interpret all features in the SAE.

        Args:
            texts: Text corpus for analysis.
            k: Number of examples per feature.
            save_path: Optional path to save results.

        Returns:
            Dictionary mapping feature_id -> interpretation.
        """
        interpretations = {}

        print(f"\nInterpreting {self.sae.d_hidden} features...")

        for feature_id in tqdm(range(self.sae.d_hidden)):
            interp = self.interpret_feature(feature_id, texts, k=k)
            interpretations[feature_id] = interp

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(interpretations, f, indent=2)

            print(f"Saved interpretations to {save_path}")

        # Print summary
        active_features = sum(1 for v in interpretations.values() if v['active'])
        print(f"\nSummary:")
        print(f"  Active features: {active_features} / {self.sae.d_hidden}")
        print(f"  Dead features: {self.sae.d_hidden - active_features}")

        return interpretations


def analyze_sae(
    sae_checkpoint_path: str,
    model_wrapper,
    layer_idx: int,
    texts: List[str],
    save_dir: str = './results/interpretations',
    device: str = 'cuda'
) -> Dict:
    """
    Convenience function to analyze a trained SAE.

    Args:
        sae_checkpoint_path: Path to SAE checkpoint.
        model_wrapper: GPT2WithResidualHooks instance.
        layer_idx: Layer index.
        texts: Text corpus for analysis.
        save_dir: Directory to save results.
        device: Device to use.

    Returns:
        Feature interpretations.
    """
    from models import JumpReLUSAE

    # Load checkpoint
    print(f"Loading SAE checkpoint from {sae_checkpoint_path}...")
    checkpoint = torch.load(sae_checkpoint_path, map_location=device)

    # Create SAE
    # Get d_model and d_hidden from checkpoint
    state_dict = checkpoint['model_state_dict']
    d_model = state_dict['W_enc.weight'].shape[1]
    d_hidden = state_dict['W_enc.weight'].shape[0]

    sae = JumpReLUSAE(
        d_model=d_model,
        d_hidden=d_hidden
    )
    sae.load_state_dict(state_dict)

    print(f"Loaded SAE: d_model={d_model}, d_hidden={d_hidden}")
    print(f"Best loss: {checkpoint['best_loss']:.6f}")

    # Create interpreter
    interpreter = FeatureInterpreter(
        sae=sae,
        model_wrapper=model_wrapper,
        layer_idx=layer_idx,
        device=device
    )

    # Interpret features
    save_path = Path(save_dir) / f'layer_{layer_idx}_interpretations.json'
    interpretations = interpreter.interpret_all_features(
        texts=texts,
        k=20,
        save_path=str(save_path)
    )

    return interpretations


def print_feature_summary(interpretation: Dict):
    """
    Print human-readable summary of a feature.

    Args:
        interpretation: Feature interpretation dict.
    """
    feature_id = interpretation['feature_id']

    print(f"\nFeature {feature_id}")
    print("=" * 60)

    if not interpretation['active']:
        print("  Status: DEAD (never activated)")
        return

    print(f"  Status: ACTIVE")
    print(f"  Top activation: {interpretation['top_activation']:.4f}")
    print(f"  Mean activation: {interpretation['mean_activation']:.4f}")
    print(f"  Num activations: {interpretation['num_activations']}")

    print(f"\n  Top tokens:")
    for token, count in interpretation['common_tokens'][:10]:
        print(f"    '{token}': {count}")

    print(f"\n  Top example:")
    if interpretation['top_examples']:
        ex = interpretation['top_examples'][0]
        print(f"    Text: {ex['text']}...")
        print(f"    Position: {ex['position']}")
        print(f"    Activation: {ex['activation']:.4f}")
