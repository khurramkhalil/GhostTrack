"""
run_mechanism_study.py — Main cluster entry point for Phase 2 experiments.

Pipeline phases (controlled by --phase):

  data      Load HaluEval / TruthfulQA, save HallucinationDataset to disk.
  sae       Extract Wikipedia hidden states and train one SAE per layer.
  track     Run the hypothesis tracker over the labelled dataset.
  detect    Fit / evaluate the HallucinationDetector on tracking metrics.

Usage (local test, CPU):
    python scripts/run_mechanism_study.py \\
        --model gpt2 \\
        --config config/model_configs/gpt2-medium.yaml \\
        --phase data \\
        --output-dir /tmp/ghosttrack_test

Usage (cluster, all phases):
    python scripts/run_mechanism_study.py \\
        --model gpt2-medium \\
        --config config/model_configs/gpt2-medium.yaml \\
        --phase all \\
        --output-dir /data/experiments/gpt2-medium \\
        --data-dir /data/datasets \\
        --sae-checkpoint-dir /data/sae_checkpoints/gpt2-medium \\
        --seed 42 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Utility: ensure package is importable when running without install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GhostTrack mechanism study pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Model name or HF path (e.g. gpt2-medium, Qwen/Qwen2.5-1.5B)")
    p.add_argument("--config", default=None,
                   help="Path to YAML config (auto-detected from --model if omitted)")
    p.add_argument("--phase", choices=["data", "sae", "track", "detect", "all"],
                   default="all", help="Which pipeline phase to run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--output-dir", default="experiments/run_001",
                   help="Root directory for all outputs")
    p.add_argument("--data-dir", default=None,
                   help="Dataset cache directory (default: /data/datasets on k8s, else ./data)")
    p.add_argument("--sae-checkpoint-dir", default=None,
                   help="Where to save/load SAE checkpoints")
    p.add_argument("--halueval-split", default="qa",
                   choices=["qa", "summarization", "dialogue"],
                   help="HaluEval subset to use")
    p.add_argument("--n-tokens", type=int, default=1_000_000,
                   help="Wikipedia tokens per SAE layer (sae phase)")
    p.add_argument("--batch-size-extract", type=int, default=8)
    p.add_argument("--batch-size-sae", type=int, default=1024)
    p.add_argument("--sae-epochs", type=int, default=None,
                   help="Override SAE training epochs from config")
    p.add_argument("--top-k-features", type=int, default=50,
                   help="Top-k SAE features per layer for tracking")
    p.add_argument("--detector-type", default="random_forest",
                   choices=["random_forest", "gradient_boosting",
                            "logistic_regression", "svm", "ensemble"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def phase_data(args, cfg, out_dir: Path, data_dir: Path):
    """Download and save labelled hallucination dataset."""
    from ghosttrack.data import HaluEvalLoader

    print(f"\n[data] Loading HaluEval split={args.halueval_split} ...")
    loader = HaluEvalLoader(cache_dir=str(data_dir))
    dataset = loader.load(split=args.halueval_split)

    n_factual = dataset.n_factual
    n_halluc = dataset.n_hallucinated
    print(f"[data] Loaded {len(dataset)} examples  "
          f"({n_factual} factual, {n_halluc} hallucinated)")

    ds_path = out_dir / "dataset.json"
    dataset.save(str(ds_path))
    print(f"[data] Saved to {ds_path}")
    return dataset


def phase_sae(args, cfg, out_dir: Path, data_dir: Path, ckpt_dir: Path, model):
    """Extract Wikipedia hidden states and train one SAE per layer."""
    import torch
    from ghosttrack.data import WikipediaCorpus, HiddenStateExtractor, load_hidden_states
    from ghosttrack.sae import JumpReLUSAE, SAETrainer
    from types import SimpleNamespace

    model_id = args.model.replace("/", "-")
    corpus = WikipediaCorpus(cache_dir=str(data_dir))
    extractor = HiddenStateExtractor(model, corpus,
                                     cache_dir=str(data_dir / "hidden_states" / model_id))

    sae_cfg = SimpleNamespace(
        learning_rate=cfg.sae_training.learning_rate,
        weight_decay=cfg.sae_training.weight_decay,
        gradient_clip=cfg.sae_training.gradient_clip,
        epochs=args.sae_epochs or cfg.sae_training.epochs,
        batch_size=args.batch_size_sae,
        val_split=0.05,
    )

    trainer = SAETrainer(device=args.device)

    # Single forward pass collects ALL layers simultaneously — avoids N separate
    # passes through the corpus (which causes GPU to sit idle between batches).
    print(f"\n[sae] Extracting {args.n_tokens:,} tokens for all "
          f"{model.n_layers} layers (single pass) ...")
    all_paths = extractor.extract_all_layers_single_pass(
        num_tokens=args.n_tokens,
        batch_size=args.batch_size_extract,
        flush_every=5,
    )

    for layer_idx in range(model.n_layers):
        # Layer-adaptive lambda_sparse: deep layers have richer representations
        # and need more active features; excessive sparsity penalty causes
        # divergence (observed in GPT-2 Medium L22-23).  Linear decay from
        # base value at layer 0 down to 30% of base at the final layer.
        _decay = max(0.3, 1.0 - 0.7 * layer_idx / max(model.n_layers - 1, 1))
        lambda_sparse = cfg.sae.lambda_sparse * _decay

        print(f"\n[sae] Training layer {layer_idx}/{model.n_layers - 1}  "
              f"(lambda_sparse={lambda_sparse:.3f})")
        hidden_states = load_hidden_states(str(all_paths[layer_idx]))

        sae = JumpReLUSAE(
            d_model=cfg.sae.d_model or model.d_model,
            d_hidden=cfg.sae.d_hidden,
            threshold=cfg.sae.threshold,
            lambda_sparse=lambda_sparse,
        )
        history = trainer.train(
            sae, hidden_states, sae_cfg,
            layer_idx=layer_idx,
            save_dir=str(ckpt_dir),
        )
        print(f"[sae] Layer {layer_idx} done. "
              f"Best val loss: {min(history['val_loss']):.6f}")

        # Free memory between layers
        del sae, hidden_states
        torch.cuda.empty_cache() if args.device == "cuda" else None


def phase_track(args, cfg, out_dir: Path, ckpt_dir: Path, model, dataset):
    """Run hypothesis tracker over the dataset and save metrics."""
    import gc
    import numpy as np
    import torch
    from ghosttrack.tracking import FeatureExtractor, HypothesisTracker
    from ghosttrack.metrics import MetricsRegistry

    print(f"\n[track] Building FeatureExtractor from {ckpt_dir} ...")
    extractor = FeatureExtractor.from_checkpoints(model, str(ckpt_dir),
                                                  device=args.device)
    registry = MetricsRegistry()
    n_layers = model.n_layers

    features_list = []
    labels = []

    for i, ex in enumerate(dataset):
        if i % 50 == 0:
            print(f"[track] {i}/{len(dataset)}")
        # Periodically force GC to release GPU/CPU tensors from prior iterations.
        if i % 500 == 0 and i > 0:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        text = ex.prompt + " " + ex.completion

        layer_features = None
        tracker = None
        try:
            layer_features = extractor.extract(text, max_length=256)

            tracker = HypothesisTracker(config={
                "birth_threshold": cfg.tracking.birth_threshold,
                "association_threshold": cfg.tracking.association_threshold,
                "semantic_weight": cfg.tracking.semantic_weight,
                "activation_weight": cfg.tracking.activation_weight,
            })

            # Build layer feature lists for tracker
            top_k = args.top_k_features
            l0_feats = extractor.get_top_k_features(layer_features[0], k=top_k)
            tracker.initialize(l0_feats)
            for lf in layer_features[1:]:
                feats = extractor.get_top_k_features(lf, k=top_k)
                tracker.step(lf.layer, feats)
            tracker.finalize()

            vec = registry.feature_vector(tracker, n_layers)
            features_list.append(vec)
            labels.append(ex.label)
        except Exception as exc:
            print(f"[track] Skipping example {ex.id}: {exc}")
        finally:
            # Explicitly release per-example tensors so GPU/CPU memory is
            # returned promptly rather than waiting for the next GC cycle.
            del layer_features, tracker

    X = np.stack(features_list)
    y = np.array(labels, dtype=int)

    np.save(out_dir / "features.npy", X)
    np.save(out_dir / "labels.npy", y)
    print(f"[track] Saved features {X.shape} and labels {y.shape}")
    return X, y


def phase_detect(args, cfg, out_dir: Path, X=None, y=None):
    """Fit detector and evaluate on held-out test split."""
    import numpy as np
    from ghosttrack.detection import HallucinationDetector, DetectionMetrics

    if X is None:
        X = np.load(out_dir / "features.npy")
        y = np.load(out_dir / "labels.npy")

    # Simple 80/20 split (multi-seed cross-val is done in analysis notebooks)
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    det = HallucinationDetector(model_type=args.detector_type,
                                n_layers=cfg.model.n_layers)
    det.fit(X_train, y_train)
    metrics = det.evaluate(X_test, y_test)

    result = {
        "auroc": metrics.auroc,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "n_train": n_train,
        "n_test": len(X_test),
        "detector_type": args.detector_type,
    }

    print("\n[detect] Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    results_path = out_dir / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[detect] Saved results to {results_path}")

    det.save(str(out_dir / "detector.pkl"))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- Paths ----
    import os
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else (
        Path("/data/datasets") if os.path.isdir("/data") else Path("./data/cache")
    )
    ckpt_dir = Path(args.sae_checkpoint_dir) if args.sae_checkpoint_dir else (
        out_dir / "sae_checkpoints"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Config ----
    from ghosttrack.utils import load_config, set_seed, get_logger, save_config

    logger = get_logger("ghosttrack.run")
    set_seed(args.seed)

    if args.config:
        cfg = load_config(args.config)
    else:
        # Auto-detect config from model name
        guesses = [
            f"config/model_configs/{args.model.replace('/', '-')}.yaml",
            f"config/model_configs/{args.model.split('/')[-1]}.yaml",
        ]
        cfg = None
        for g in guesses:
            p = _REPO_ROOT / g
            if p.exists():
                cfg = load_config(str(p))
                break
        if cfg is None:
            from ghosttrack.utils import Config
            cfg = Config()
            logger.warning("No config found; using defaults.")

    # Save config snapshot
    save_config(cfg, str(out_dir / "config_snapshot.yaml"))
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Model: {args.model}  Phase: {args.phase}")

    # ---- Model (only needed for sae / track phases) ----
    phases = [args.phase] if args.phase != "all" else ["data", "sae", "track", "detect"]
    model = None

    if any(ph in phases for ph in ("sae", "track")):
        from ghosttrack.models import create_model
        logger.info(f"Loading model: {args.model}")
        model = create_model(args.model, device=args.device)
        logger.info(f"Model: {model.n_layers} layers, d_model={model.d_model}")

    # ---- Execute phases ----
    dataset = None
    X = y = None

    if "data" in phases:
        dataset = phase_data(args, cfg, out_dir, data_dir)

    if "sae" in phases:
        phase_sae(args, cfg, out_dir, data_dir, ckpt_dir, model)

    if "track" in phases:
        if dataset is None:
            from ghosttrack.data import HallucinationDataset
            ds_path = out_dir / "dataset.json"
            if not ds_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {ds_path}.  "
                    "Run with --phase data first."
                )
            dataset = HallucinationDataset.load(str(ds_path))
        X, y = phase_track(args, cfg, out_dir, ckpt_dir, model, dataset)

    if "detect" in phases:
        phase_detect(args, cfg, out_dir, X=X, y=y)

    logger.info("All phases complete.")


if __name__ == "__main__":
    main()
