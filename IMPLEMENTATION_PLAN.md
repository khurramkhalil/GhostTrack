# GhostTrack: Multi-Hypothesis Tracking for Hallucination Detection
## Complete Implementation Plan

---

## **PROJECT OVERVIEW**

### Objective
Build an end-to-end system that detects hallucinations in LLM outputs by tracking competing semantic hypotheses across transformer layers using Sparse Autoencoders (SAEs) and multi-object tracking techniques.

### Core Innovation
Unlike existing methods (LSD), this system provides **interpretable explanations** by visualizing how competing semantic tracks emerge, compete, and resolve into either factual or hallucinated outputs.

### Target Performance
- **Minimum**: AUROC ≥ 0.90 on TruthfulQA
- **Target**: AUROC ≥ 0.92 with interpretability
- **Stretch**: AUROC > 0.94 on multiple datasets

---

## **SYSTEM ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT TEXT                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  GPT-2 Model    │
         │  with Hooks     │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼────┐  ┌────▼────┐
│Residual│  │   MLP   │  │  Attn   │
│ Stream │  │ Outputs │  │ Outputs │
└───┬────┘  └────┬────┘  └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │  12 JumpReLU    │
         │  SAE Encoders   │
         │  (per layer)    │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼────┐  ┌────▼────┐
│Feature │  │Reconst. │  │Feature  │
│Activat.│  │ Error   │  │Embeddin.│
└───┬────┘  └────┬────┘  └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │   Hypothesis    │
         │    Tracker      │
         │  (Birth/Death)  │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼────┐  ┌────▼────┐
│Entropy │  │ Track   │  │Competit.│
│Metric  │  │ Churn   │  │ Score   │
└───┬────┘  └────┬────┘  └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │  Hallucination  │
         │    Detector     │
         │  (Ensemble)     │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼────┐  ┌────▼────┐
│Predict.│  │ Visual. │  │Explanat.│
└────────┘  └─────────┘  └─────────┘
```

---

## **PHASE 1: INFRASTRUCTURE SETUP**
**Timeline**: Week 1 (7 days)
**Goal**: Build foundation for data loading, model instrumentation, and basic pipelines

### 1.1 Environment Setup
**File**: `requirements.txt`

```bash
# Core ML frameworks
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0
plotly>=5.14.0

# Optimization
wandb>=0.15.0
optuna>=3.2.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
```

**Tasks**:
- [ ] Create virtual environment
- [ ] Install all dependencies
- [ ] Verify CUDA/GPU availability
- [ ] Set up wandb for experiment tracking
- [ ] Create project directory structure

### 1.2 Project Structure
```
GhostTrack/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── model_config.yaml
│   ├── sae_config.yaml
│   └── training_config.yaml
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── dataset_generator.py
│   └── cache/
├── models/
│   ├── __init__.py
│   ├── model_wrapper.py
│   ├── sae_model.py
│   └── checkpoints/
├── tracking/
│   ├── __init__.py
│   ├── feature_extractor.py
│   ├── hypothesis_tracker.py
│   ├── track_association.py
│   └── track.py
├── detection/
│   ├── __init__.py
│   ├── detector.py
│   ├── divergence_metrics.py
│   └── classifiers.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── ablations.py
│   └── metrics.py
├── visualization/
│   ├── __init__.py
│   ├── visualize.py
│   ├── case_studies.py
│   └── plots.py
├── scripts/
│   ├── train_saes.py
│   ├── run_detection.py
│   ├── generate_visualizations.py
│   └── run_ablations.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sae_training.ipynb
│   ├── 03_tracking_demo.ipynb
│   └── 04_results_analysis.ipynb
└── tests/
    ├── test_data_loader.py
    ├── test_sae.py
    ├── test_tracker.py
    └── test_detector.py
```

### 1.3 Data Pipeline
**File**: `data/data_loader.py`

**Implementation Details**:
```python
class HallucinationDataset:
    """
    Loads and processes TruthfulQA dataset.
    Creates factual/hallucinated pairs.
    """

    def __init__(self, dataset_name='truthful_qa', cache_dir='./data/cache'):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def load_truthfulqa(self):
        """
        Load TruthfulQA from HuggingFace
        Format: {question, best_answer, incorrect_answers, category}
        """
        pass

    def create_pairs(self):
        """
        Create factual/hallucinated pairs:
        - Factual: question + best_answer
        - Hallucinated: question + incorrect_answer
        """
        pass

    def split_data(self, train=0.7, val=0.15, test=0.15):
        """
        Split into train/val/test sets
        Stratified by category
        """
        pass

    def get_item(self, idx):
        return {
            'prompt': str,              # Input question
            'factual_answer': str,      # Ground truth
            'hallucinated_answer': str, # Incorrect answer
            'category': str,            # Question category
            'metadata': dict            # Additional info
        }
```

**Tasks**:
- [ ] Implement TruthfulQA loader
- [ ] Create factual/hallucinated pairs
- [ ] Implement data splitting (stratified by category)
- [ ] Add data validation checks
- [ ] Create data statistics report
- [ ] Cache processed datasets

### 1.4 Model Wrapper with Residual Stream Hooks
**File**: `models/model_wrapper.py`

**Critical Requirements**:
- Hook residual stream (post-attention + post-MLP)
- Hook MLP outputs separately
- Hook attention outputs separately

**Implementation Details**:
```python
class GPT2WithResidualHooks:
    """
    Wrapper around GPT-2 that extracts:
    1. Residual stream (full block output)
    2. MLP outputs (separately)
    3. Attention outputs (separately)
    """

    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.hooks = []
        self.cache = {
            'residual_stream': [],
            'mlp_outputs': [],
            'attn_outputs': []
        }

    def register_hooks(self):
        """
        Register forward hooks on:
        - transformer.h[i] (full residual)
        - transformer.h[i].mlp (MLP only)
        - transformer.h[i].attn (attention only)
        """
        pass

    def forward_with_cache(self, input_ids, attention_mask=None):
        """
        Returns:
        {
            'logits': tensor [batch, seq_len, vocab],
            'residual_stream': List[tensor],  # 12 x [batch, seq_len, 768]
            'mlp_outputs': List[tensor],      # 12 x [batch, seq_len, 768]
            'attn_outputs': List[tensor],     # 12 x [batch, seq_len, 768]
        }
        """
        pass

    def clear_cache(self):
        """Reset cache for new forward pass"""
        pass
```

**Tasks**:
- [ ] Implement GPT2WithResidualHooks class
- [ ] Add hook registration for all components
- [ ] Test forward pass with cache
- [ ] Validate output shapes
- [ ] Add batch processing support
- [ ] Create unit tests

### 1.5 Validation Tests
**File**: `tests/test_infrastructure.py`

**Tests to implement**:
- [ ] Data loader returns correct format
- [ ] Model hooks capture all required tensors
- [ ] Batch processing works correctly
- [ ] Memory usage is reasonable
- [ ] Cache clearing works properly

**Deliverables**:
- ✅ Functional data pipeline
- ✅ Instrumented GPT-2 model
- ✅ Project structure created
- ✅ All tests passing

---

## **PHASE 2: JUMPRELU SAE TRAINING**
**Timeline**: Week 2-3 (14 days)
**Goal**: Train 12 high-quality SAEs (one per layer)

### 2.1 JumpReLU SAE Architecture
**File**: `models/sae_model.py`

**Implementation Details**:
```python
class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder
    Better reconstruction-sparsity tradeoff than standard ReLU
    """

    def __init__(self, d_model=768, d_hidden=4096, threshold=0.1, lambda_sparse=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.lambda_sparse = lambda_sparse

        # Encoder and decoder
        self.W_enc = nn.Linear(d_model, d_hidden, bias=True)
        self.W_dec = nn.Linear(d_hidden, d_model, bias=True)

        # Initialize decoder as normalized
        self.normalize_decoder()

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm"""
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(
                self.W_dec.weight.data, dim=0
            )

    def encode(self, x):
        """JumpReLU activation with learned threshold"""
        pre_activation = self.W_enc(x)
        # JumpReLU: x if x > threshold, else 0
        mask = (pre_activation > self.threshold).float()
        return pre_activation * mask

    def decode(self, encoded):
        """Decode features back to original space"""
        return self.W_dec(encoded)

    def forward(self, x):
        """
        Forward pass with reconstruction and error tracking
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        error = x - decoded

        return {
            'reconstruction': decoded,
            'features': encoded,
            'error': error,  # Hallucinations may hide here
            'sparsity': (encoded > 0).float().mean()
        }

    def loss(self, x):
        """
        Combined reconstruction + sparsity loss
        """
        output = self.forward(x)

        # MSE reconstruction loss
        recon_loss = F.mse_loss(output['reconstruction'], x)

        # L1 sparsity loss
        sparsity_loss = output['features'].abs().mean()

        # Total loss
        total_loss = recon_loss + self.lambda_sparse * sparsity_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'sparsity': output['sparsity']
        }
```

**Tasks**:
- [ ] Implement JumpReLU activation
- [ ] Implement encoder/decoder
- [ ] Add loss computation
- [ ] Add decoder normalization
- [ ] Create unit tests
- [ ] Validate forward/backward pass

### 2.2 SAE Training Pipeline
**File**: `scripts/train_saes.py`

**Training Configuration**:
```yaml
# config/sae_config.yaml
model:
  d_model: 768
  d_hidden: 4096
  threshold: 0.1
  lambda_sparse: 0.01

training:
  epochs: 20
  batch_size: 256
  learning_rate: 1e-4
  weight_decay: 0.0
  gradient_clip: 1.0

data:
  source: "wikipedia"
  num_tokens: 100_000_000
  max_length: 512

validation:
  every_n_steps: 1000
  target_recon_loss: 0.01
  target_sparsity: 50-100  # active features per token
```

**Implementation Details**:
```python
def train_sae_for_layer(
    layer_idx: int,
    model_wrapper: GPT2WithResidualHooks,
    config: dict,
    device: str = 'cuda'
):
    """
    Train SAE for a specific layer
    """

    # Initialize SAE
    sae = JumpReLUSAE(
        d_model=config['model']['d_model'],
        d_hidden=config['model']['d_hidden'],
        threshold=config['model']['threshold'],
        lambda_sparse=config['model']['lambda_sparse']
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(sae, dataloader, optimizer, device)
        val_metrics = validate(sae, val_dataloader, device)

        # Log to wandb
        wandb.log({
            f'layer_{layer_idx}/train_loss': train_loss,
            f'layer_{layer_idx}/val_recon_loss': val_metrics['recon_loss'],
            f'layer_{layer_idx}/val_sparsity': val_metrics['sparsity'],
            f'layer_{layer_idx}/learning_rate': scheduler.get_last_lr()[0]
        })

        # Checkpoint
        if val_metrics['recon_loss'] < best_loss:
            save_checkpoint(sae, layer_idx, val_metrics)

        scheduler.step()

    return sae

def extract_hidden_states(
    model_wrapper: GPT2WithResidualHooks,
    dataset: Dataset,
    layer_idx: int
):
    """
    Extract hidden states from specific layer for SAE training
    """
    pass

def train_all_saes(config_path: str):
    """
    Train SAEs for all 12 layers in parallel (if resources allow)
    """
    pass
```

**Tasks**:
- [ ] Implement hidden state extraction
- [ ] Implement training loop
- [ ] Add validation metrics
- [ ] Add checkpointing
- [ ] Add wandb logging
- [ ] Implement parallel training for multiple layers
- [ ] Create training monitoring dashboard

**Target Metrics**:
- Reconstruction loss < 0.01
- Active features: 50-100 per token
- Training time: ~4-6 hours per layer (A100)

### 2.3 Feature Interpretation + Error Analysis
**File**: `evaluation/interpret_features.py`

**Implementation Details**:
```python
def interpret_sae_features(
    sae: JumpReLUSAE,
    layer_idx: int,
    dataset: Dataset,
    top_k: int = 20
):
    """
    Interpret SAE features by finding top-activating examples
    """

    feature_interpretations = {}

    for feature_id in range(sae.d_hidden):
        # Find top-activating examples
        top_examples = find_top_activating(
            sae, feature_id, dataset, k=top_k
        )

        # Extract common tokens
        common_tokens = extract_common_tokens(top_examples)

        # Semantic labeling (manual or automated)
        semantic_label = label_feature(common_tokens, top_examples)

        feature_interpretations[feature_id] = {
            'label': semantic_label,
            'top_tokens': common_tokens,
            'top_examples': top_examples,
            'avg_activation': compute_avg_activation(sae, feature_id, dataset)
        }

    # NEW: Analyze reconstruction errors
    error_patterns = analyze_reconstruction_errors(sae, dataset)

    return {
        'feature_labels': feature_interpretations,
        'error_patterns': error_patterns
    }

def analyze_reconstruction_errors(sae: JumpReLUSAE, dataset: Dataset):
    """
    Analyze patterns in reconstruction errors
    May correlate with hallucination locations
    """
    pass
```

**Tasks**:
- [ ] Implement top-activating example finder
- [ ] Create automatic semantic labeling
- [ ] Implement error pattern analysis
- [ ] Create feature interpretation report
- [ ] Visualize feature activations
- [ ] Manual review and refinement of labels

**Deliverables**:
- ✅ 12 trained SAEs (one per layer)
- ✅ Validation metrics all meet targets
- ✅ Feature interpretation labels for all features
- ✅ Error pattern analysis complete

---

## **PHASE 3: HYPOTHESIS TRACKING SYSTEM**
**Timeline**: Week 3-4 (7 days)
**Goal**: Build the core tracking system

### 3.1 Track Data Structure
**File**: `tracking/track.py`

**Implementation Details**:
```python
@dataclass
class Track:
    """
    Represents a semantic hypothesis tracked across layers
    """
    track_id: int
    feature_embedding: np.ndarray  # Semantic representation
    birth_layer: int
    token_pos: int
    death_layer: Optional[int] = None
    trajectory: List[Tuple[int, float, np.ndarray]] = field(default_factory=list)
    # trajectory: [(layer, activation, embedding), ...]

    def update(self, layer: int, activation: float, embedding: np.ndarray):
        """Add new observation to trajectory"""
        self.trajectory.append((layer, activation, embedding))

    def is_alive(self) -> bool:
        """Check if track is still active"""
        return self.death_layer is None

    def get_activation_at(self, layer: int) -> Optional[float]:
        """Get activation value at specific layer"""
        for l, act, _ in self.trajectory:
            if l == layer:
                return act
        return None

    def layer_range(self) -> range:
        """Get range of layers this track exists in"""
        start = self.birth_layer
        end = self.death_layer if self.death_layer else 12
        return range(start, end)

    def max_activation(self) -> float:
        """Get maximum activation across trajectory"""
        return max([act for _, act, _ in self.trajectory])

    def final_activation(self) -> float:
        """Get final activation value"""
        return self.trajectory[-1][1] if self.trajectory else 0.0
```

### 3.2 Feature Extractor
**File**: `tracking/feature_extractor.py`

**Implementation Details**:
```python
class LayerwiseFeatureExtractor:
    """
    Extracts SAE features for each layer
    """

    def __init__(
        self,
        model_wrapper: GPT2WithResidualHooks,
        saes: List[JumpReLUSAE]
    ):
        self.model = model_wrapper
        self.saes = saes  # 12 SAEs, one per layer

    def extract_features(self, text: str) -> List[Dict]:
        """
        Extract features and errors for all layers

        Returns:
        List of dicts with keys:
        - layer: int
        - features: tensor [seq_len, 4096]
        - error: tensor [seq_len, 768]
        - mlp_contribution: tensor [seq_len, 768]
        """

        # Tokenize
        inputs = self.model.tokenizer(
            text, return_tensors='pt'
        ).to(self.model.device)

        # Forward pass with hooks
        outputs = self.model.forward_with_cache(
            inputs['input_ids'],
            inputs['attention_mask']
        )

        layer_features = []

        for layer_idx in range(12):
            # Get hidden states
            hidden = outputs['residual_stream'][layer_idx]  # [batch, seq, 768]

            # Pass through SAE
            with torch.no_grad():
                sae_output = self.saes[layer_idx].forward(hidden)

            layer_features.append({
                'layer': layer_idx,
                'features': sae_output['features'],         # [seq, 4096]
                'error': sae_output['error'],               # [seq, 768]
                'mlp_contribution': outputs['mlp_outputs'][layer_idx],
                'hidden_state': hidden
            })

        return layer_features

    def get_top_k_features(
        self,
        features: torch.Tensor,
        k: int = 50,
        token_pos: Optional[int] = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Get top-k activated features at a position

        Returns:
        List of (feature_id, activation, embedding)
        """
        pass
```

**Tasks**:
- [ ] Implement feature extraction pipeline
- [ ] Add top-k feature selection
- [ ] Implement feature embedding extraction
- [ ] Add batch processing support
- [ ] Create unit tests

### 3.3 Track Association Algorithm
**File**: `tracking/track_association.py`

**Critical**: Use semantic similarity, NOT feature IDs

**Implementation Details**:
```python
def associate_features_between_layers(
    prev_tracks: List[Track],
    curr_features: List[Tuple[int, float, np.ndarray]],
    layer_idx: int,
    config: dict
) -> Tuple[List[Tuple[Track, Tuple]], List[Tuple]]:
    """
    Associate features between layers using semantic similarity

    Returns:
    - associations: List[(track, feature)]
    - unmatched_features: List[feature]
    """

    # Filter alive tracks
    alive_tracks = [t for t in prev_tracks if t.is_alive()]

    if len(alive_tracks) == 0 or len(curr_features) == 0:
        return [], curr_features

    # Build cost matrix
    n_tracks = len(alive_tracks)
    n_features = len(curr_features)
    cost_matrix = np.zeros((n_tracks, n_features))

    for i, track in enumerate(alive_tracks):
        # Get last embedding from track
        last_embedding = track.trajectory[-1][2]  # [4096]
        last_activation = track.trajectory[-1][1]

        for j, (feat_id, feat_act, feat_emb) in enumerate(curr_features):
            # Cost component 1: Semantic distance (cosine)
            semantic_cost = 1 - cosine_similarity(
                last_embedding.reshape(1, -1),
                feat_emb.reshape(1, -1)
            )[0, 0]

            # Cost component 2: Activation change
            activation_cost = abs(last_activation - feat_act) / (last_activation + 1e-6)

            # Cost component 3: Spatial proximity (token position)
            position_cost = abs(track.token_pos - feat_id) / 100  # Normalize

            # Weighted combination
            cost_matrix[i, j] = (
                config['semantic_weight'] * semantic_cost +
                config['activation_weight'] * activation_cost +
                config['position_weight'] * position_cost
            )

    # Hungarian algorithm for optimal assignment
    from scipy.optimize import linear_sum_assignment
    track_indices, feature_indices = linear_sum_assignment(cost_matrix)

    # Filter by threshold
    associations = []
    matched_features = set()

    for i, j in zip(track_indices, feature_indices):
        if cost_matrix[i, j] < config['association_threshold']:
            associations.append((alive_tracks[i], curr_features[j]))
            matched_features.add(j)

    # Find unmatched features
    unmatched_features = [
        curr_features[j] for j in range(n_features)
        if j not in matched_features
    ]

    return associations, unmatched_features

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
```

**Configuration**:
```yaml
# config/tracking_config.yaml
association:
  semantic_weight: 0.6
  activation_weight: 0.2
  position_weight: 0.2
  association_threshold: 0.5
  birth_threshold: 0.5
  death_threshold: 0.1
  top_k_features: 50
```

**Tasks**:
- [ ] Implement cost matrix computation
- [ ] Integrate Hungarian algorithm
- [ ] Add threshold filtering
- [ ] Create unit tests with synthetic data
- [ ] Validate on real examples

### 3.4 Hypothesis Tracker
**File**: `tracking/hypothesis_tracker.py`

**Implementation Details**:
```python
class HypothesisTracker:
    """
    Manages track lifecycle across all layers
    """

    def __init__(self, config: dict):
        self.config = config
        self.tracks: List[Track] = []
        self.track_id_counter = 0

    def initialize_tracks(self, layer_0_features: List[Tuple[int, float, np.ndarray]]):
        """
        Create initial tracks from layer 0 top features
        """
        for feat_id, activation, embedding in layer_0_features:
            if activation > self.config['birth_threshold']:
                track = Track(
                    track_id=self.track_id_counter,
                    feature_embedding=embedding,
                    birth_layer=0,
                    token_pos=feat_id
                )
                track.update(0, activation, embedding)
                self.tracks.append(track)
                self.track_id_counter += 1

    def update_tracks(
        self,
        layer_idx: int,
        current_features: List[Tuple[int, float, np.ndarray]]
    ):
        """
        Update existing tracks and create new ones
        """
        # Associate current features with existing tracks
        associations, unmatched_features = associate_features_between_layers(
            self.tracks,
            current_features,
            layer_idx,
            self.config
        )

        # Update matched tracks
        matched_tracks = set()
        for track, (feat_id, activation, embedding) in associations:
            track.update(layer_idx, activation, embedding)
            matched_tracks.add(track.track_id)

        # Mark unmatched alive tracks as dead
        for track in self.tracks:
            if track.is_alive() and track.track_id not in matched_tracks:
                # Check if it was active in previous layer
                if layer_idx - 1 >= track.birth_layer:
                    track.death_layer = layer_idx - 1

        # Create new tracks for unmatched features
        for feat_id, activation, embedding in unmatched_features:
            if activation > self.config['birth_threshold']:
                new_track = Track(
                    track_id=self.track_id_counter,
                    feature_embedding=embedding,
                    birth_layer=layer_idx,
                    token_pos=feat_id
                )
                new_track.update(layer_idx, activation, embedding)
                self.tracks.append(new_track)
                self.track_id_counter += 1

    def get_alive_tracks(self, layer_idx: Optional[int] = None) -> List[Track]:
        """Get all tracks alive at specified layer"""
        if layer_idx is None:
            return [t for t in self.tracks if t.is_alive()]
        return [
            t for t in self.tracks
            if t.birth_layer <= layer_idx and
               (t.death_layer is None or t.death_layer >= layer_idx)
        ]

    def reset(self):
        """Reset tracker for new example"""
        self.tracks = []
        self.track_id_counter = 0
```

**Tasks**:
- [ ] Implement track initialization
- [ ] Implement track update logic
- [ ] Add track death detection
- [ ] Create comprehensive unit tests
- [ ] Test on sample examples
- [ ] Validate track trajectories make sense

**Deliverables**:
- ✅ Feature extraction working for all layers
- ✅ Track association algorithm validated
- ✅ Hypothesis tracker fully functional
- ✅ Unit tests passing
- ✅ Sample tracks visualized and validated

---

## **PHASE 4: HALLUCINATION DETECTION PIPELINE**
**Timeline**: Week 4-5 (7 days)
**Goal**: Build detection model and achieve target AUROC

### 4.1 Divergence Metrics
**File**: `detection/divergence_metrics.py`

**Implementation Details**:
```python
def compute_track_divergence(tracks: List[Track]) -> Dict[str, Any]:
    """
    Compute comprehensive metrics characterizing track evolution
    """

    metrics = {}

    # 1. HYPOTHESIS ENTROPY (KEY METRIC)
    entropy_per_layer = []
    for layer in range(12):
        active_tracks = [
            t for t in tracks
            if layer in t.layer_range()
        ]

        if len(active_tracks) == 0:
            entropy_per_layer.append(0.0)
            continue

        # Get activations at this layer
        activations = np.array([
            t.get_activation_at(layer) for t in active_tracks
        ])

        # Normalize to probability distribution
        probs = softmax(activations)

        # Compute Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_per_layer.append(entropy)

    metrics['entropy_trajectory'] = entropy_per_layer
    metrics['max_entropy'] = max(entropy_per_layer)
    metrics['entropy_peak_layer'] = int(np.argmax(entropy_per_layer))
    metrics['entropy_final'] = entropy_per_layer[-1]

    # Expected pattern:
    # - Factual: Entropy decreases monotonically
    # - Hallucination: Entropy spikes mid-layers (competition)

    # 2. BIRTH/DEATH EVENTS
    birth_layers = [t.birth_layer for t in tracks]
    death_layers = [t.death_layer for t in tracks if t.death_layer is not None]

    metrics['total_tracks'] = len(tracks)
    metrics['birth_death_ratio'] = len(death_layers) / (len(tracks) + 1e-10)
    metrics['avg_birth_layer'] = np.mean(birth_layers) if birth_layers else 0
    metrics['avg_death_layer'] = np.mean(death_layers) if death_layers else 0

    # 3. TRACK STABILITY
    trajectory_variances = []
    for track in tracks:
        if len(track.trajectory) > 2:
            activations = [act for _, act, _ in track.trajectory]
            trajectory_variances.append(np.var(activations))

    metrics['mean_trajectory_variance'] = np.mean(trajectory_variances) if trajectory_variances else 0

    # 4. COMPETITION SCORE
    # Count layers with >2 high-activation competing tracks
    competition_layers = 0
    for layer in range(12):
        active_at_layer = [
            t.get_activation_at(layer)
            for t in tracks if t.get_activation_at(layer) is not None
        ]
        high_activation = [a for a in active_at_layer if a > 0.5]
        if len(high_activation) > 2:
            competition_layers += 1

    metrics['competition_score'] = competition_layers / 12

    # 5. WINNER DOMINANCE
    # Check if one track dominates in final layers
    final_tracks = [
        t for t in tracks
        if t.death_layer is None or t.death_layer >= 10
    ]

    if final_tracks:
        final_activations = [t.final_activation() for t in final_tracks]
        metrics['winner_dominance'] = max(final_activations) / (sum(final_activations) + 1e-10)
        metrics['num_final_tracks'] = len(final_tracks)
    else:
        metrics['winner_dominance'] = 0.0
        metrics['num_final_tracks'] = 0

    # 6. ENTROPY TREND
    # Fit linear regression to entropy trajectory
    if len(entropy_per_layer) > 3:
        from scipy.stats import linregress
        x = np.arange(len(entropy_per_layer))
        slope, intercept, r_value, _, _ = linregress(x, entropy_per_layer)
        metrics['entropy_slope'] = slope
        metrics['entropy_r_squared'] = r_value ** 2
    else:
        metrics['entropy_slope'] = 0.0
        metrics['entropy_r_squared'] = 0.0

    return metrics

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
```

**Tasks**:
- [ ] Implement all metric computations
- [ ] Add entropy calculation
- [ ] Add competition detection
- [ ] Validate metrics on sample data
- [ ] Create visualization for each metric

### 4.2 Hallucination Detector
**File**: `detection/detector.py`

**Implementation Details**:
```python
class HallucinationDetector:
    """
    End-to-end hallucination detection system
    """

    def __init__(
        self,
        model_wrapper: GPT2WithResidualHooks,
        saes: List[JumpReLUSAE],
        config: dict
    ):
        self.model = model_wrapper
        self.saes = saes
        self.config = config

        # Components
        self.feature_extractor = LayerwiseFeatureExtractor(model_wrapper, saes)
        self.tracker = HypothesisTracker(config['tracking'])

        # Classifier (lightweight)
        self.classifier = None  # Will be trained in Phase 4.3

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text contains hallucination

        Returns:
        {
            'is_hallucination': bool,
            'confidence': float,
            'metrics': dict,
            'tracks': List[Track],
            'explanation': dict
        }
        """

        # Reset tracker
        self.tracker.reset()

        # Extract features for all layers
        layer_features = self.feature_extractor.extract_features(text)

        # Initialize tracks from layer 0
        top_features_l0 = self.feature_extractor.get_top_k_features(
            layer_features[0]['features'],
            k=self.config['tracking']['top_k_features']
        )
        self.tracker.initialize_tracks(top_features_l0)

        # Update tracks through all layers
        for layer_idx in range(1, 12):
            top_features = self.feature_extractor.get_top_k_features(
                layer_features[layer_idx]['features'],
                k=self.config['tracking']['top_k_features']
            )
            self.tracker.update_tracks(layer_idx, top_features)

        # Compute divergence metrics
        metrics = compute_track_divergence(self.tracker.tracks)

        # Classification strategies
        scores = self._compute_scores(metrics)

        # Ensemble final score
        final_score = (
            self.config['detection']['entropy_weight'] * scores['entropy'] +
            self.config['detection']['churn_weight'] * scores['churn'] +
            self.config['detection']['ml_weight'] * scores['ml']
        )

        return {
            'is_hallucination': final_score > self.config['detection']['threshold'],
            'confidence': final_score,
            'metrics': metrics,
            'tracks': self.tracker.tracks,
            'explanation': self._generate_explanation(metrics, self.tracker.tracks)
        }

    def _compute_scores(self, metrics: dict) -> dict:
        """Compute multiple hallucination scores"""

        scores = {}

        # Strategy 1: Entropy threshold
        scores['entropy'] = float(
            metrics['max_entropy'] > self.config['detection']['entropy_threshold']
        )

        # Strategy 2: Birth/death ratio
        scores['churn'] = float(
            metrics['birth_death_ratio'] > self.config['detection']['churn_threshold']
        )

        # Strategy 3: ML classifier (if trained)
        if self.classifier is not None:
            feature_vector = self._extract_feature_vector(metrics)
            scores['ml'] = self.classifier.predict_proba([feature_vector])[0][1]
        else:
            scores['ml'] = 0.0

        return scores

    def _extract_feature_vector(self, metrics: dict) -> np.ndarray:
        """Extract feature vector for ML classifier"""
        return np.array([
            metrics['max_entropy'],
            metrics['entropy_peak_layer'],
            metrics['birth_death_ratio'],
            metrics['mean_trajectory_variance'],
            metrics['competition_score'],
            metrics['winner_dominance'],
            metrics['entropy_slope'],
            metrics['num_final_tracks']
        ])

    def _generate_explanation(
        self,
        metrics: dict,
        tracks: List[Track]
    ) -> dict:
        """Generate human-readable explanation"""

        critical_layer = metrics['entropy_peak_layer']

        # Find tracks active at critical layer
        active_at_critical = [
            t for t in tracks
            if critical_layer in t.layer_range()
        ]

        # Sort by activation at critical layer
        active_at_critical.sort(
            key=lambda t: t.get_activation_at(critical_layer),
            reverse=True
        )

        # Find winner
        winner = max(tracks, key=lambda t: t.final_activation()) if tracks else None

        return {
            'summary': f"Competition detected at layer {critical_layer}",
            'critical_layer': critical_layer,
            'competing_tracks': active_at_critical[:3],  # Top 3
            'winner': winner,
            'max_entropy': metrics['max_entropy'],
            'num_tracks': len(tracks)
        }
```

**Configuration**:
```yaml
# config/detection_config.yaml
detection:
  entropy_threshold: 1.5
  churn_threshold: 0.3
  entropy_weight: 0.4
  churn_weight: 0.3
  ml_weight: 0.3
  threshold: 0.5
```

**Tasks**:
- [ ] Implement detector class
- [ ] Implement scoring strategies
- [ ] Add explanation generation
- [ ] Create unit tests
- [ ] Test on sample examples

### 4.3 Training ML Classifier
**File**: `detection/classifiers.py`

**Implementation Details**:
```python
def train_classifier(
    detector: HallucinationDetector,
    train_dataset: HallucinationDataset
) -> sklearn.base.BaseEstimator:
    """
    Train lightweight classifier on divergence metrics
    """

    X_train = []
    y_train = []

    # Collect features from training data
    for example in train_dataset:
        # Factual
        result_factual = detector.predict(example['factual_answer'])
        X_train.append(detector._extract_feature_vector(result_factual['metrics']))
        y_train.append(0)

        # Hallucinated
        result_halluc = detector.predict(example['hallucinated_answer'])
        X_train.append(detector._extract_feature_vector(result_halluc['metrics']))
        y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train classifier (logistic regression or random forest)
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    classifier.fit(X_train, y_train)

    return classifier
```

### 4.4 Evaluation Pipeline
**File**: `evaluation/evaluate.py`

**Implementation Details**:
```python
def evaluate_on_truthfulqa(
    detector: HallucinationDetector,
    dataset: HallucinationDataset,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on TruthfulQA
    """

    results = []

    for example in tqdm(dataset, desc="Evaluating"):
        # Predict on factual answer
        pred_factual = detector.predict(example['factual_answer'])

        # Predict on hallucinated answer
        pred_halluc = detector.predict(example['hallucinated_answer'])

        results.append({
            'example_id': example['id'],
            'category': example['category'],
            'factual_score': pred_factual['confidence'],
            'halluc_score': pred_halluc['confidence'],
            'factual_metrics': pred_factual['metrics'],
            'halluc_metrics': pred_halluc['metrics'],
            'factual_tracks': pred_factual['tracks'],
            'halluc_tracks': pred_halluc['tracks']
        })

    # Compute metrics
    y_true = [0] * len(results) + [1] * len(results)
    y_scores = [r['factual_score'] for r in results] + [r['halluc_score'] for r in results]
    y_pred = [s > 0.5 for s in y_scores]

    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )

    evaluation_metrics = {
        'auroc': roc_auc_score(y_true, y_scores),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred)
    }

    # Per-category analysis
    category_metrics = analyze_by_category(results, y_true, y_scores)

    if save_results:
        save_evaluation_results(results, evaluation_metrics, category_metrics)

    return {
        'metrics': evaluation_metrics,
        'category_metrics': category_metrics,
        'results': results
    }

def analyze_by_category(results, y_true, y_scores):
    """Analyze performance by question category"""
    pass
```

**Tasks**:
- [ ] Implement evaluation loop
- [ ] Add metric computation
- [ ] Add per-category analysis
- [ ] Create evaluation report
- [ ] Tune hyperparameters on validation set
- [ ] Achieve target AUROC ≥ 0.90

**Deliverables**:
- ✅ Detector achieving AUROC ≥ 0.90
- ✅ All metrics computed and logged
- ✅ Per-category analysis complete
- ✅ Evaluation report generated

---

## **PHASE 5: VISUALIZATION & CASE STUDIES**
**Timeline**: Week 5-6 (7 days)
**Goal**: Create compelling visualizations and interpretable case studies

### 5.1 Semantic Radar Plot
**File**: `visualization/visualize.py`

**Implementation** (see PRD for full code)

**Tasks**:
- [ ] Implement UMAP projection
- [ ] Create semantic radar plot
- [ ] Add track trajectory overlay
- [ ] Color code by track fate
- [ ] Add annotations for birth/death events

### 5.2 Track Trajectory Plot
**File**: `visualization/visualize.py`

**Tasks**:
- [ ] Implement activation vs layer plot
- [ ] Color code by track fate
- [ ] Add birth/death markers
- [ ] Create side-by-side factual vs hallucination plots

### 5.3 Entropy Trajectory Plot
**File**: `visualization/plots.py`

**Implementation**:
```python
def plot_entropy_trajectory(
    factual_metrics: dict,
    halluc_metrics: dict,
    title: str = "Hypothesis Entropy Comparison"
):
    """
    Compare entropy trajectories for factual vs hallucinated
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    layers = range(12)

    # Factual entropy
    ax.plot(
        layers,
        factual_metrics['entropy_trajectory'],
        marker='o',
        linewidth=2,
        color='green',
        label='Factual Answer'
    )

    # Hallucinated entropy
    ax.plot(
        layers,
        halluc_metrics['entropy_trajectory'],
        marker='x',
        linewidth=2,
        color='red',
        label='Hallucinated Answer'
    )

    # Mark peak layers
    ax.axvline(
        factual_metrics['entropy_peak_layer'],
        linestyle='--',
        color='green',
        alpha=0.5
    )
    ax.axvline(
        halluc_metrics['entropy_peak_layer'],
        linestyle='--',
        color='red',
        alpha=0.5
    )

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Hypothesis Entropy', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
```

### 5.4 Case Study Generator
**File**: `visualization/case_studies.py`

**Implementation** (see PRD for full code)

**Tasks**:
- [ ] Implement case study report generator
- [ ] Create HTML visualization dashboard
- [ ] Generate 20+ diverse case studies
- [ ] Include all visualizations
- [ ] Add narrative explanations

**Deliverables**:
- ✅ Semantic radar plots for all test examples
- ✅ Track trajectory visualizations
- ✅ 20+ detailed case studies with narratives
- ✅ Interactive HTML dashboard

---

## **PHASE 6: OPTIMIZATION & ABLATIONS**
**Timeline**: Week 6-7 (7 days)
**Goal**: Validate design decisions and optimize performance

### 6.1 Hyperparameter Tuning
**File**: `scripts/tune_hyperparameters.py`

**Implementation**:
```python
import optuna

def objective(trial):
    """Optuna objective function"""

    # Suggest hyperparameters
    config = {
        'tracking': {
            'semantic_weight': trial.suggest_float('semantic_weight', 0.3, 0.8),
            'activation_weight': trial.suggest_float('activation_weight', 0.1, 0.4),
            'position_weight': trial.suggest_float('position_weight', 0.1, 0.4),
            'association_threshold': trial.suggest_float('association_threshold', 0.3, 0.7),
            'birth_threshold': trial.suggest_float('birth_threshold', 0.3, 0.7),
            'top_k_features': trial.suggest_int('top_k_features', 30, 100)
        },
        'detection': {
            'entropy_threshold': trial.suggest_float('entropy_threshold', 1.0, 2.5),
            'churn_threshold': trial.suggest_float('churn_threshold', 0.2, 0.5),
            'entropy_weight': trial.suggest_float('entropy_weight', 0.2, 0.6),
            'churn_weight': trial.suggest_float('churn_weight', 0.2, 0.4),
            'ml_weight': trial.suggest_float('ml_weight', 0.2, 0.4)
        }
    }

    # Train with these parameters
    detector = HallucinationDetector(model, saes, config)
    results = evaluate_on_truthfulqa(detector, val_dataset)

    return results['metrics']['auroc']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best AUROC: {study.best_value}")
print(f"Best params: {study.best_params}")
```

**Tasks**:
- [ ] Set up Optuna study
- [ ] Define search space
- [ ] Run optimization on validation set
- [ ] Analyze parameter importance
- [ ] Update configs with best parameters

### 6.2 Ablation Studies
**File**: `evaluation/ablations.py`

**Implementation**:
```python
ablation_studies = {
    'single_hypothesis': {
        'name': 'Single Hypothesis (No Competition)',
        'modify': lambda detector: setattr(
            detector.config['tracking'], 'top_k_features', 1
        ),
        'purpose': 'Validate that multi-hypothesis tracking adds value'
    },
    'no_association': {
        'name': 'No Track Association',
        'modify': lambda detector: disable_association(detector),
        'purpose': 'Validate that temporal tracking matters'
    },
    'feature_id_association': {
        'name': 'Feature ID Association',
        'modify': lambda detector: use_feature_id_matching(detector),
        'purpose': 'Validate critique about semantic similarity'
    },
    'standard_relu': {
        'name': 'Standard ReLU SAE',
        'modify': lambda detector: replace_with_relu_sae(detector),
        'purpose': 'Validate that JumpReLU improves results'
    },
    'no_error_term': {
        'name': 'Ignore Reconstruction Error',
        'modify': lambda detector: disable_error_tracking(detector),
        'purpose': 'Check if hallucinations hide in reconstruction residuals'
    }
}

def run_ablation_study(
    baseline_detector: HallucinationDetector,
    test_dataset: HallucinationDataset
) -> Dict[str, float]:
    """Run all ablations and compare to baseline"""

    # Baseline performance
    baseline_results = evaluate_on_truthfulqa(baseline_detector, test_dataset)
    baseline_auroc = baseline_results['metrics']['auroc']

    ablation_results = {
        'baseline': {'auroc': baseline_auroc, 'delta': 0.0}
    }

    for ablation_id, ablation in ablation_studies.items():
        print(f"\n Running ablation: {ablation['name']}")

        # Create modified detector
        modified_detector = copy.deepcopy(baseline_detector)
        ablation['modify'](modified_detector)

        # Evaluate
        results = evaluate_on_truthfulqa(modified_detector, test_dataset)
        auroc = results['metrics']['auroc']
        delta = baseline_auroc - auroc

        ablation_results[ablation_id] = {
            'name': ablation['name'],
            'auroc': auroc,
            'delta': delta,
            'purpose': ablation['purpose']
        }

        print(f"  AUROC: {auroc:.4f} (Δ = {delta:.4f})")

    return ablation_results
```

**Tasks**:
- [ ] Implement all 5 ablations
- [ ] Run on test set
- [ ] Analyze performance drops
- [ ] Create ablation report
- [ ] Validate all design decisions

**Deliverables**:
- ✅ Optimized hyperparameters
- ✅ All ablations complete
- ✅ Performance analysis report
- ✅ Design validation complete

---

## **PHASE 7: PAPER WRITING & CODE RELEASE**
**Timeline**: Week 7-8 (7 days)
**Goal**: Complete paper and release code

### 7.1 Paper Writing
**File**: `paper/main.tex`

**Sections**:
1. Abstract (250 words)
2. Introduction (2 pages)
3. Background (2 pages)
4. Method (4 pages)
5. Experiments (3 pages)
6. Case Studies (2 pages)
7. Ablations & Analysis (2 pages)
8. Discussion (1 page)
9. Conclusion (0.5 page)

**Tasks**:
- [ ] Write first draft of all sections
- [ ] Create all figures (high quality)
- [ ] Create all tables
- [ ] Write supplementary material
- [ ] Proofread and revise
- [ ] Get feedback from collaborators
- [ ] Final polish

### 7.2 Code Release
**Repository**: `github.com/ghosttrack/multi-hypothesis-tracking`

**Tasks**:
- [ ] Clean up all code
- [ ] Add comprehensive docstrings
- [ ] Create detailed README
- [ ] Write installation guide
- [ ] Create usage examples
- [ ] Add Jupyter notebook demos
- [ ] Release pretrained SAEs
- [ ] Create documentation website
- [ ] Add license (MIT)
- [ ] Test on clean environment

### 7.3 Documentation
**Files**:
- README.md (comprehensive)
- INSTALL.md
- USAGE.md
- API.md
- CONTRIBUTING.md

**Tasks**:
- [ ] Write installation instructions
- [ ] Create usage examples
- [ ] Document all APIs
- [ ] Add troubleshooting guide
- [ ] Create FAQ

**Deliverables**:
- ✅ Complete paper draft
- ✅ All figures publication-ready
- ✅ Code repository public
- ✅ Pretrained models released
- ✅ Documentation complete

---

## **SUCCESS CRITERIA**

### Minimum Viable (Conference Acceptance)
- ✅ AUROC ≥ 0.90 on TruthfulQA
- ✅ 10 clear case studies showing track competition
- ✅ Semantic radar visualization working
- ✅ At least 3 successful ablations

### Strong Contribution (Spotlight/Oral)
- ✅ AUROC ≥ 0.92
- ✅ 20 case studies with diverse error types
- ✅ Leading indicator validated (2-3 layer advantage)
- ✅ All 5 ablations successful
- ✅ Code + pretrained SAEs released

### Exceptional (Best Paper Contender)
- ✅ AUROC > 0.94
- ✅ Works on multiple models (GPT-2, GPT-2-medium, GPT-2-large)
- ✅ Validated on multiple datasets (TruthfulQA + HaluEval)
- ✅ Theoretical analysis of why entropy predicts hallucinations

---

## **RISK MITIGATION**

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SAEs don't learn interpretable features | Medium | High | Use JumpReLU, increase sparsity, try different architectures |
| Semantic association fails | Low | Critical | Validate on synthetic data first, tune cost function |
| AUROC < 0.90 | Medium | High | Add ensemble methods, try different classifiers |
| No clear track competition | Low | Medium | Select different question types, focus on factual recall |
| Can't beat LSD performance | High | Medium | Emphasize interpretability, not raw performance |
| Training takes too long | Medium | Medium | Use smaller model (GPT-2), reduce dataset size |
| GPU resource constraints | Low | High | Use cloud GPUs (Lambda Labs, RunPod) |

---

## **TIMELINE SUMMARY**

```
Week 1: Infrastructure Setup
├─ Data pipeline
├─ Model wrapper
└─ Project structure

Week 2-3: SAE Training
├─ JumpReLU implementation
├─ Train 12 SAEs
└─ Feature interpretation

Week 3-4: Tracking System
├─ Feature extraction
├─ Track association
└─ Hypothesis tracker

Week 4-5: Detection Pipeline
├─ Divergence metrics
├─ Detector implementation
└─ Evaluation (AUROC ≥ 0.90)

Week 5-6: Visualization
├─ Semantic radar plots
├─ Trajectory plots
└─ 20+ case studies

Week 6-7: Optimization
├─ Hyperparameter tuning
└─ Ablation studies

Week 7-8: Paper Writing
├─ Draft complete paper
├─ Code release
└─ Documentation
```

---

## **FINAL CHECKLIST**

### Before Starting
- [ ] Confirm GPU access (A100 or equivalent)
- [ ] Download TruthfulQA dataset
- [ ] Install all dependencies
- [ ] Verify GPT-2 loads correctly
- [ ] Set up wandb account

### Critical Validations
- [ ] JumpReLU SAE reconstruction loss < 0.01
- [ ] Feature interpretation makes semantic sense
- [ ] Track association produces stable trajectories
- [ ] Entropy metric shows expected patterns
- [ ] AUROC ≥ 0.90 on validation set

### Paper Submission
- [ ] All figures publication-ready
- [ ] Code repository public + documented
- [ ] Pretrained SAEs available for download
- [ ] Reproducibility verified on clean machine
- [ ] All ablations completed
- [ ] Supplementary material complete

---

## **CONCLUSION**

This implementation plan provides a complete roadmap for building **GhostTrack**, a novel system for interpretable hallucination detection using multi-hypothesis tracking. The plan is structured to:

1. **Build incrementally** - Each phase builds on the previous
2. **Validate continuously** - Tests and validations at each step
3. **Achieve targets** - Clear success criteria and metrics
4. **Mitigate risks** - Identified risks with mitigation strategies
5. **Deliver value** - Both research contribution and practical tool

The end result will be:
- A high-performing hallucination detector (AUROC ≥ 0.90)
- Interpretable explanations via track visualization
- Open-source code and pretrained models
- A publishable research paper
- A valuable tool for the community

**Total estimated time**: 8 weeks (56 days) for complete end-to-end system.
