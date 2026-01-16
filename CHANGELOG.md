# Changelog

All notable changes to GhostTrack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-XX

### Added
- Initial release of GhostTrack
- Multi-hypothesis tracking system with semantic similarity-based association
- Sparse Autoencoder (SAE) implementation with JumpReLU activation
- 26 divergence metrics across 6 families (Entropy, Churn, Competition, Stability, Dominance, Density)
- Multiple detector models (Random Forest, Gradient Boosting, Logistic Regression, SVM, Ensemble)
- Complete visualization suite:
  - Track trajectory plots
  - Competition heatmaps
  - Divergence metric visualizations
  - Activation timelines
- Interactive HTML dashboards with Plotly
- Case study generation system
- Hyperparameter tuning with GridSearchCV
- Ablation study framework
- Comprehensive test suite (180+ tests)
- Documentation and examples

### Phase 1: Infrastructure
- Configuration management system
- TruthfulQA dataset loader with train/val/test splits
- GPT-2 model wrapper with residual stream hooks
- SAE model with learned thresholds

### Phase 2: SAE Training
- Wikipedia corpus loader for training data
- Hidden state extraction pipeline
- SAE training with cosine annealing + gradient clipping
- Feature interpretation tools
- Checkpoint management system

### Phase 3: Hypothesis Tracking
- Track dataclass for semantic hypotheses
- Layerwise feature extractor
- Semantic association using Hungarian algorithm
- Hypothesis tracker for lifecycle management
- Birth/update/death mechanics

### Phase 4: Hallucination Detection
- 26 divergence metrics implementation
- 5 classifier types with ensemble support
- Feature scaling and model serialization
- Feature importance analysis
- End-to-end evaluation pipeline on TruthfulQA

### Phase 5: Visualization & Case Studies
- Track trajectory visualization
- Competition heatmap generation
- Interactive dashboard with Plotly
- Detailed case study generation
- Batch processing for multiple studies

### Phase 6: Optimization & Ablation
- Hyperparameter tuning framework
- Grid search with cross-validation
- Metric family ablation studies
- Individual feature ablation
- Cumulative feature addition analysis
- Automated report generation

### Phase 7: Research & Release
- Complete research paper draft
- Comprehensive documentation
- Contributing guidelines
- MIT License
- Changelog

## [0.1.0] - Development

### Added
- Core infrastructure and proof of concept

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

No migration needed for new users. For development users:

1. Update dependencies: `pip install -r requirements.txt`
2. Re-train SAEs with new JumpReLU implementation
3. Update tracking configuration to use semantic weights
4. Regenerate divergence metrics with new families

---

## Future Releases

### [1.1.0] - Planned

- Support for larger models (GPT-3 scale)
- Real-time streaming detection
- Multi-modal hallucination detection
- Cross-lingual evaluation
- Model intervention capabilities

### [2.0.0] - Vision

- Theoretical framework formalization
- Causal intervention experiments
- Production deployment tools
- API service for hallucination detection

---

For detailed changes, see individual commit messages and pull requests.
