"""
Main script to extract hidden states and train SAEs.

This orchestrates the full Phase 2 pipeline:
1. Load Wikipedia corpus
2. Extract hidden states for all layers
3. Train SAEs for each layer
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models import GPT2WithResidualHooks
from data.wikipedia_loader import WikipediaCorpus, HiddenStateExtractor
from train_sae import train_sae_for_layer
from config import load_config


def main():
    parser = argparse.ArgumentParser(
        description='Extract hidden states and train SAEs'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract hidden states, do not train'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train SAEs (assumes states already extracted)'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='all',
        help='Layers to process (e.g., "0,3,6" or "all")'
    )
    parser.add_argument(
        '--num-tokens',
        type=int,
        default=10_000_000,
        help='Number of tokens to extract per layer'
    )
    parser.add_argument(
        '--batch-size-extract',
        type=int,
        default=8,
        help='Batch size for extraction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./data/cache',
        help='Cache directory'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./models/checkpoints',
        help='Directory to save trained models'
    )

    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Determine layers to process
    if args.layers == 'all':
        layers = list(range(config.model.n_layers))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    print(f"\nProcessing layers: {layers}")
    print(f"Device: {args.device}")
    print(f"Num tokens per layer: {args.num_tokens:,}\n")

    # Phase 1: Extract Hidden States
    if not args.train_only:
        print("=" * 70)
        print("PHASE 1: EXTRACTING HIDDEN STATES")
        print("=" * 70)

        # Load model
        print("\nLoading GPT-2 model...")
        model = GPT2WithResidualHooks(
            model_name=config.model.base_model,
            device=args.device
        )

        # Load Wikipedia corpus
        print("Loading Wikipedia corpus...")
        corpus = WikipediaCorpus(
            cache_dir=args.cache_dir,
            max_tokens=args.num_tokens * len(layers)
        )
        corpus.load()

        # Create extractor
        extractor = HiddenStateExtractor(
            model_wrapper=model,
            corpus=corpus,
            cache_dir=f"{args.cache_dir}/hidden_states"
        )

        # Extract for each layer
        for layer_idx in layers:
            print(f"\n{'='*70}")
            print(f"Extracting layer {layer_idx}/{config.model.n_layers-1}")
            print(f"{'='*70}")

            state_path = extractor.extract_for_layer(
                layer_idx=layer_idx,
                num_tokens=args.num_tokens,
                batch_size=args.batch_size_extract,
                max_length=512
            )

            print(f"✓ Saved to {state_path}")

        print(f"\n{'='*70}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*70}\n")

    # Phase 2: Train SAEs
    if not args.extract_only:
        print("=" * 70)
        print("PHASE 2: TRAINING SAEs")
        print("=" * 70)

        cache_dir = Path(args.cache_dir) / 'hidden_states'

        for layer_idx in layers:
            print(f"\n{'='*70}")
            print(f"Training SAE for layer {layer_idx}/{config.model.n_layers-1}")
            print(f"{'='*70}")

            # Path to hidden states
            states_path = cache_dir / f'layer_{layer_idx}_states.pt'

            if not states_path.exists():
                print(f"ERROR: Hidden states not found at {states_path}")
                print("Run with --extract-only first or without --train-only")
                continue

            # Train
            try:
                history = train_sae_for_layer(
                    layer_idx=layer_idx,
                    hidden_states_path=str(states_path),
                    config_path=args.config,
                    save_dir=args.save_dir,
                    device=args.device
                )

                print(f"\n✓ Layer {layer_idx} training complete")
                print(f"  Best recon loss: {min(history['recon_loss']):.6f}")

            except Exception as e:
                print(f"\n✗ Error training layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}\n")

    print("\nAll tasks complete!")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
