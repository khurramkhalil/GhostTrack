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

    def find_top_activating_examples(
        self,
        feature_id: int,
        texts: List[str],
        k: int = 20,
        max_length: int = 512
    ) -> List[Tuple[str, int, float]]:
        """
        Find top-k examples that activate a specific feature.

        Args:
            feature_id: Feature ID to analyze.
            texts: List of text examples to search.
            k: Number of top examples to return.
            max_length: Max sequence length.

        Returns:
            List of (text, token_position, activation_value) tuples.
        """
        top_activations = []  # (text, position, activation)

        with torch.no_grad():
            for text in tqdm(texts, desc=f"Feature {feature_id}"):
                # Get model activations
                outputs = self.model.process_text(text, max_length=max_length)
                layer_states = outputs['residual_stream'][self.layer_idx]

                # Pass through SAE
                sae_output = self.sae.forward(layer_states)
                features = sae_output['features']  # [1, seq_len, d_hidden]

                # Get activations for this feature
                feature_acts = features[0, :, feature_id].cpu().numpy()

                # Find max activation in this text
                max_pos = np.argmax(feature_acts)
                max_act = feature_acts[max_pos]

                if max_act > 0:  # Only if feature is active
                    top_activations.append((text, int(max_pos), float(max_act)))

        # Sort by activation and take top-k
        top_activations.sort(key=lambda x: x[2], reverse=True)
        return top_activations[:k]

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
            feature_id, texts, k=k
        )

        if not top_examples:
            return {
                'feature_id': feature_id,
                'active': False,
                'top_activation': 0.0
            }

        # Extract common tokens
        common_tokens = self.extract_common_tokens(top_examples)

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
            'top_activation': top_examples[0][2],
            'mean_activation': np.mean([act for _, _, act in top_examples]),
            'common_tokens': sorted_tokens,
            'top_examples': [
                {
                    'text': text[:200],  # Truncate for storage
                    'position': pos,
                    'activation': act
                }
                for text, pos, act in top_examples[:5]
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
