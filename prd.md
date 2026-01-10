AUTHORITATIVE DESIGN DOCUMENT
Multi-Hypothesis Tracking for Hallucination Interpretability
Version 2.0 - Post-Critique Revision
PHASE 1: INFRASTRUCTURE SETUP (Week 1)
Step 1.1: Environment Setup
# Core dependencies
pip install torch>=2.0 transformers>=4.30 datasets
pip install scikit-learn numpy pandas matplotlib seaborn
pip install umap-learn tqdm wandb  # Added: UMAP for visualization
Step 1.2: Data Pipeline
File: data_loader.py

"""
Load TruthfulQA + generate hallucination pairs
Primary: TruthfulQA
Secondary (if time): HaluEval, MemoTrap
"""

class HallucinationDataset:
    def __init__(self, dataset_name='truthful_qa'):
        # Load dataset
        # Create factual/hallucinated pairs
        # Split: 70/15/15 train/val/test
        
    def get_item(self, idx):
        return {
            'prompt': str,
            'factual_answer': str,
            'hallucinated_answer': str,
            'category': str,
            'metadata': dict
        }
Step 1.3: Model Wrapper with Residual Stream Hooks
File: model_wrapper.py
CRITICAL UPDATE: Hook both residual stream AND MLP outputs separately

class GPT2WithResidualHooks:
    """
    Extract:
    1. Residual stream (post-attention + post-MLP)
    2. MLP outputs (separately)
    3. Attention outputs (separately)
    
    Rationale: Factual tracks live in residual stream,
               hallucination distractors emerge from MLPs
    """
    
    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.hooks = []
        
    def forward_with_cache(self, input_ids):
        """
        Returns:
        {
            'logits': tensor,
            'residual_stream': List[tensor],  # Post full block
            'mlp_outputs': List[tensor],      # MLP only
            'attn_outputs': List[tensor],     # Attention only
        }
        """
        # Implementation with hooks on:
        # - transformer.h[i] (full residual)
        # - transformer.h[i].mlp (MLP output)
        # - transformer.h[i].attn (attention output)
PHASE 2: JUMPRELU SAE TRAINING (Week 2-3)
Step 2.1: JumpReLU SAE Architecture
File: sae_model.py
CRITICAL UPDATE: Use JumpReLU instead of standard ReLU

class JumpReLUSAE(nn.Module):
    """
    Architecture from Rajamanoharan et al., 2024
    Better reconstruction-sparsity trade-off
    """
    
    def __init__(self, d_model=768, d_hidden=4096, threshold=0.1):
        self.W_enc = nn.Linear(d_model, d_hidden, bias=True)
        self.W_dec = nn.Linear(d_hidden, d_model, bias=True)
        self.threshold = threshold
        
    def encode(self, x):
        """JumpReLU activation with learned threshold"""
        pre_activation = self.W_enc(x)
        # JumpReLU: x if x > threshold, else 0
        mask = (pre_activation > self.threshold).float()
        return pre_activation * mask
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.W_dec(encoded)
        
        # Track reconstruction error (CRITICAL ADDITION)
        error = x - decoded
        
        return {
            'reconstruction': decoded,
            'features': encoded,
            'error': error  # Hallucinations may hide here
        }
        
    def loss(self, x):
        output = self.forward(x)
        recon_loss = F.mse_loss(output['reconstruction'], x)
        sparsity_loss = output['features'].abs().mean()
        return recon_loss + self.lambda_sparse * sparsity_loss
Step 2.2: SAE Training Pipeline
File: train_sae.py

"""
Train 12 JumpReLU SAEs (one per layer)
Data: Wikipedia clean text (100M tokens)
"""

def train_sae_for_layer(layer_idx, hidden_states_dataset):
    sae = JumpReLUSAE(d_model=768, d_hidden=4096)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(20):
        for batch in dataloader:
            loss = sae.loss(batch)
            loss.backward()
            optimizer.step()
            
    # Validation
    val_recon_loss = evaluate_reconstruction(sae, val_data)
    val_sparsity = compute_sparsity(sae, val_data)
    
    # Save with metadata
    torch.save({
        'model_state': sae.state_dict(),
        'layer': layer_idx,
        'recon_loss': val_recon_loss,
        'sparsity': val_sparsity
    }, f'checkpoints/sae_layer_{layer_idx}.pt')
Target Metrics:

Reconstruction loss < 0.01
Active features: 50-100 per token
Training time: ~4-6 hours per layer (A100)
Step 2.3: Feature Interpretation + Error Analysis
File: interpret_features.py
CRITICAL ADDITION: Also analyze reconstruction errors

def interpret_sae_features(sae, layer_idx, dataset):
    """
    For each feature:
    1. Find top-activating examples
    2. Extract common tokens
    3. Manual semantic labeling
    """
    feature_labels = {}
    
    for feature_id in range(4096):
        top_examples = find_top_activating(sae, feature_id, dataset)
        semantic_label = manually_label(top_examples)
        feature_labels[feature_id] = semantic_label
        
    # NEW: Analyze error patterns
    error_patterns = analyze_reconstruction_errors(sae, dataset)
    
    return {
        'feature_labels': feature_labels,
        'error_patterns': error_patterns  # May correlate with hallucinations
    }
PHASE 3: HYPOTHESIS TRACKING SYSTEM (Week 3-4)
Step 3.1: Feature Extraction Per Layer
File: feature_extractor.py

class LayerwiseFeatureExtractor:
    def __init__(self, model, saes):
        self.model = model
        self.saes = saes  # List of 12 SAEs
        
    def extract_features(self, text):
        """
        Extract features + residual errors per layer
        """
        outputs = self.model.forward_with_cache(text)
        
        layer_features = []
        for layer_idx in range(12):
            hidden = outputs['residual_stream'][layer_idx]
            sae_output = self.saes[layer_idx].forward(hidden)
            
            layer_features.append({
                'layer': layer_idx,
                'features': sae_output['features'],  # [seq_len, 4096]
                'error': sae_output['error'],        # Reconstruction residual
                'mlp_contribution': outputs['mlp_outputs'][layer_idx]
            })
            
        return layer_features
Step 3.2: Track Management Logic
File: hypothesis_tracker.py

class Track:
    """
    Represents a semantic hypothesis across layers
    """
    def __init__(self, feature_embedding, birth_layer, token_pos):
        self.feature_embedding = feature_embedding  # Semantic representation
        self.birth_layer = birth_layer
        self.death_layer = None
        self.trajectory = []  # [(layer, activation, embedding)]
        self.confidence_history = []
        
    def update(self, layer, activation, embedding):
        self.trajectory.append((layer, activation, embedding))
        
    def is_alive(self):
        return self.death_layer is None

class HypothesisTracker:
    def __init__(self):
        self.tracks = []
        self.track_id_counter = 0
        
    def initialize_tracks(self, layer_0_features):
        """Create initial tracks from layer 0 top features"""
        top_features = get_top_k_features(layer_0_features, k=50)
        
        for feat_id, activation, embedding in top_features:
            track = Track(embedding, birth_layer=0, token_pos=...)
            self.tracks.append(track)
            
    def update_tracks(self, layer_idx, current_features):
        """
        Update existing tracks + create new ones
        """
        # Match current features to existing tracks
        associations = self.associate_features(
            self.tracks, 
            current_features, 
            layer_idx
        )
        
        # Update matched tracks
        for track, matched_feature in associations:
            track.update(layer_idx, ...)
            
        # Mark unmatched tracks as dead
        for track in self.tracks:
            if track not in [t for t, _ in associations]:
                if track.is_alive():
                    track.death_layer = layer_idx - 1
                    
        # Create new tracks for unmatched features
        unmatched_features = get_unmatched(current_features, associations)
        for feature in unmatched_features:
            if feature.activation > BIRTH_THRESHOLD:
                new_track = Track(feature.embedding, layer_idx, ...)
                self.tracks.append(new_track)
Step 3.3: Track Association via Semantic Similarity
File: track_association.py
CRITICAL UPDATE: Use semantic similarity, NOT feature IDs

def associate_features_between_layers(prev_tracks, curr_features, layer_idx):
    """
    Associate using semantic similarity of feature embeddings
    
    Key insight: Features from different SAEs have no shared ID space.
    Must use semantic proximity in activation space.
    """
    
    # Build cost matrix
    n_tracks = len(prev_tracks)
    n_features = len(curr_features)
    cost_matrix = np.zeros((n_tracks, n_features))
    
    for i, track in enumerate(prev_tracks):
        if not track.is_alive():
            continue
            
        last_embedding = track.trajectory[-1][2]  # Last embedding
        
        for j, feature in enumerate(curr_features):
            curr_embedding = feature.embedding
            
            # Cost components:
            # 1. Semantic distance (cosine)
            semantic_cost = 1 - cosine_similarity(last_embedding, curr_embedding)
            
            # 2. Activation change
            last_activation = track.trajectory[-1][1]
            activation_cost = abs(last_activation - feature.activation)
            
            # 3. Spatial proximity (token position)
            position_cost = abs(track.token_pos - feature.token_pos) / seq_len
            
            # Total cost (weighted sum)
            cost_matrix[i, j] = (
                0.6 * semantic_cost + 
                0.2 * activation_cost + 
                0.2 * position_cost
            )
    
    # Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    track_indices, feature_indices = linear_sum_assignment(cost_matrix)
    
    # Filter by threshold
    associations = []
    for i, j in zip(track_indices, feature_indices):
        if cost_matrix[i, j] < ASSOCIATION_THRESHOLD:
            associations.append((prev_tracks[i], curr_features[j]))
            
    return associations
Implementation Note:

embedding = normalized SAE feature vector (not feature ID)
Precompute feature embeddings during extraction
Use cosine similarity in high-dimensional space (4096-D)
PHASE 4: HALLUCINATION DETECTION PIPELINE (Week 4-5)
Step 4.1: Track Divergence Metrics
File: divergence_metrics.py
CRITICAL ADDITIONS: Hypothesis entropy + leading indicator analysis

def compute_track_divergence(tracks):
    """
    Compute multiple metrics to characterize track evolution
    """
    
    metrics = {}
    
    # 1. Hypothesis Entropy (NEW - KEY METRIC)
    entropy_per_layer = []
    for layer in range(12):
        active_tracks = [t for t in tracks if layer in t.layer_range()]
        activations = [t.get_activation_at(layer) for t in active_tracks]
        
        # Normalize to probability distribution
        probs = softmax(activations)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_per_layer.append(entropy)
        
    metrics['entropy_trajectory'] = entropy_per_layer
    metrics['max_entropy'] = max(entropy_per_layer)
    metrics['entropy_peak_layer'] = np.argmax(entropy_per_layer)
    
    # Expected pattern:
    # Factual: Entropy decreases monotonically
    # Hallucination: Entropy spikes mid-layers (competition)
    
    # 2. Birth/Death Events
    birth_layers = [t.birth_layer for t in tracks]
    death_layers = [t.death_layer for t in tracks if t.death_layer]
    
    metrics['birth_death_ratio'] = len(death_layers) / (len(tracks) + 1e-10)
    metrics['critical_layer'] = find_highest_activity_layer(birth_layers, death_layers)
    
    # 3. Track Stability
    trajectory_variances = []
    for track in tracks:
        if len(track.trajectory) > 3:
            activations = [t[1] for t in track.trajectory]
            trajectory_variances.append(np.var(activations))
            
    metrics['mean_trajectory_variance'] = np.mean(trajectory_variances)
    
    # 4. Competition Score
    # Count layers with >2 high-activation competing tracks
    competition_layers = count_competition_layers(tracks, threshold=0.5)
    metrics['competition_score'] = competition_layers / 12
    
    # 5. Winner Emergence
    # Check if one track dominates in final layers
    final_tracks = [t for t in tracks if t.death_layer is None or t.death_layer > 10]
    if final_tracks:
        final_activations = [t.trajectory[-1][1] for t in final_tracks]
        metrics['winner_dominance'] = max(final_activations) / (sum(final_activations) + 1e-10)
    
    return metrics
Step 4.2: Detection Model
File: detector.py

class HallucinationDetector:
    def __init__(self, model, saes, tracker):
        self.model = model
        self.saes = saes
        self.tracker = tracker
        self.threshold = 0.5  # Tune on validation set
        
    def predict(self, text):
        # Extract features
        features = self.feature_extractor.extract_features(text)
        
        # Track hypotheses
        self.tracker.initialize_tracks(features[0])
        for layer_idx in range(1, 12):
            self.tracker.update_tracks(layer_idx, features[layer_idx])
            
        # Compute metrics
        metrics = compute_track_divergence(self.tracker.tracks)
        
        # Classification (multiple strategies)
        # Strategy 1: Entropy threshold
        hallucination_score_entropy = (
            metrics['max_entropy'] > ENTROPY_THRESHOLD
        )
        
        # Strategy 2: Birth/death ratio
        hallucination_score_churn = (
            metrics['birth_death_ratio'] > CHURN_THRESHOLD
        )
        
        # Strategy 3: Lightweight classifier on all metrics
        feature_vector = [
            metrics['max_entropy'],
            metrics['entropy_peak_layer'],
            metrics['birth_death_ratio'],
            metrics['mean_trajectory_variance'],
            metrics['competition_score'],
            metrics['winner_dominance']
        ]
        hallucination_score_ml = self.classifier.predict_proba(feature_vector)[1]
        
        # Ensemble
        final_score = (
            0.4 * hallucination_score_entropy +
            0.3 * hallucination_score_churn +
            0.3 * hallucination_score_ml
        )
        
        return {
            'is_hallucination': final_score > self.threshold,
            'confidence': final_score,
            'metrics': metrics,
            'tracks': self.tracker.tracks,
            'explanation': self._generate_explanation(metrics, self.tracker.tracks)
        }
        
    def _generate_explanation(self, metrics, tracks):
        """Generate human-readable explanation"""
        critical_layer = metrics['entropy_peak_layer']
        
        # Find competing tracks at critical layer
        active_at_critical = [
            t for t in tracks 
            if t.birth_layer <= critical_layer and 
               (t.death_layer is None or t.death_layer > critical_layer)
        ]
        
        return {
            'summary': f"Competition detected at layer {critical_layer}",
            'competing_tracks': active_at_critical[:3],  # Top 3
            'winner': max(tracks, key=lambda t: t.trajectory[-1][1] if t.trajectory else 0)
        }
Step 4.3: Evaluation Loop + Leading Indicator Analysis
File: evaluate.py
CRITICAL ADDITION: Validate that track events precede LSD-style drift

def evaluate_on_truthfulqa(detector, dataset, lsd_baseline=None):
    results = []
    
    for example in dataset:
        # Test factual answer
        pred_factual = detector.predict(example['factual_answer'])
        
        # Test hallucinated answer
        pred_halluc = detector.predict(example['hallucinated_answer'])
        
        results.append({
            'example_id': example['id'],
            'factual_score': pred_factual['confidence'],
            'halluc_score': pred_halluc['confidence'],
            'factual_metrics': pred_factual['metrics'],
            'halluc_metrics': pred_halluc['metrics'],
            'factual_tracks': pred_factual['tracks'],
            'halluc_tracks': pred_halluc['tracks']
        })
        
    # Compute standard metrics
    y_true = [0] * len(dataset) + [1] * len(dataset)  # 0=factual, 1=halluc
    y_scores = [r['factual_score'] for r in results] + [r['halluc_score'] for r in results]
    
    auroc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, [s > 0.5 for s in y_scores])
    
    # CRITICAL NEW ANALYSIS: Leading indicator validation
    if lsd_baseline:
        leading_indicator_analysis = compare_timing_with_lsd(results, lsd_baseline)
        # Validate: Track birth/death events occur 2-3 layers before LSD drift
        
    return {
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'results': results,
        'leading_indicator_advantage': leading_indicator_analysis if lsd_baseline else None
    }

def compare_timing_with_lsd(mht_results, lsd_results):
    """
    KEY DIFFERENTIATION ANALYSIS
    
    For each example:
    1. Find layer where MHT detects track competition (birth/death spike)
    2. Find layer where LSD detects significant drift
    3. Compute: delta_layers = lsd_layer - mht_layer
    
    Hypothesis: delta_layers should be 2-3 (MHT detects earlier)
    """
    timing_differences = []
    
    for mht_result, lsd_result in zip(mht_results, lsd_results):
        mht_critical_layer = mht_result['halluc_metrics']['entropy_peak_layer']
        lsd_drift_layer = lsd_result['drift_onset_layer']
        
        delta = lsd_drift_layer - mht_critical_layer
        timing_differences.append(delta)
        
    return {
        'mean_lead_time': np.mean(timing_differences),  # Should be 2-3
        'median_lead_time': np.median(timing_differences),
        'lead_time_distribution': timing_differences
    }
PHASE 5: VISUALIZATION & CASE STUDIES (Week 5-6)
Step 5.1: Semantic Radar Plot (CRITICAL NEW VISUALIZATION)
File: visualize.py

def plot_semantic_radar(tracks, hidden_states, title):
    """
    UMAP projection of hidden states + track trajectories overlaid
    
    This is the "money shot" figure for the paper
    """
    
    # Collect all hidden states across layers
    all_states = []
    layer_labels = []
    for layer in range(12):
        states_at_layer = hidden_states[layer]
        all_states.append(states_at_layer)
        layer_labels.extend([layer] * len(states_at_layer))
        
    all_states = np.concatenate(all_states, axis=0)
    
    # UMAP projection to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_states)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background: color by layer
    scatter = ax.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1],
        c=layer_labels, 
        cmap='viridis',
        alpha=0.3,
        s=20
    )
    
    # Overlay track trajectories
    for track in tracks:
        track_points = []
        for layer, activation, embedding in track.trajectory:
            # Project track embedding to 2D
            point_2d = reducer.transform(embedding.reshape(1, -1))
            track_points.append(point_2d[0])
            
        track_points = np.array(track_points)
        
        # Draw track as connected line
        color = 'red' if track.death_layer else 'green'
        linewidth = 3 if track.trajectory[-1][1] > 0.7 else 1
        
        ax.plot(
            track_points[:, 0], 
            track_points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.8,
            marker='o',
            markersize=5
        )
        
        # Mark birth
        ax.scatter(
            track_points[0, 0], 
            track_points[0, 1],
            color='blue',
            marker='*',
            s=200,
            label='Birth' if track == tracks[0] else None
        )
        
        # Mark death (if applicable)
        if track.death_layer:
            ax.scatter(
                track_points[-1, 0], 
                track_points[-1, 1],
                color='black',
                marker='x',
                s=200,
                label='Death' if track == tracks[0] else None
            )
            
    ax.set_title(title, fontsize=16)
    ax.legend()
    plt.colorbar(scatter, label='Layer')
    plt.tight_layout()
    
    return fig
Step 5.2: Track Trajectory Plot
File: visualize.py

def plot_track_trajectories(tracks, title):
    """
    Traditional line plot: activation vs layer
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, track in enumerate(tracks):
        layers = [t[0] for t in track.trajectory]
        activations = [t[1] for t in track.trajectory]
        
        # Color and style based on track fate
        if track.death_layer:
            color = 'red'
            linestyle = '--'
            label = f'Dead Track {i}'
        else:
            color = 'green'
            linestyle = '-'
            label = f'Surviving Track {i}'
            
        ax.plot(
            layers, 
            activations,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker='o',
            label=label
        )
        
        # Annotate birth
        ax.scatter(
            track.birth_layer, 
            track.trajectory[0][1],
            color='blue',
            marker='*',
            s=300,
            zorder=5
        )
        
        # Annotate death
        if track.death_layer:
            ax.scatter(
                track.death_layer,
                track.trajectory[-1][1],
                color='black',
                marker='x',
                s=300,
                zorder=5
            )
            
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Feature Activation', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig
Step 5.3: Case Study Generator
File: case_studies.py

def generate_case_study(example, factual_result, halluc_result):
    """
    Generate detailed narrative explanation
    """
    
    report = f"""
    ============================================
    CASE STUDY: {example['id']}
    ============================================
    
    Question: {example['prompt']}
    
    Factual Answer: {example['factual_answer']}
    Hallucinated Answer: {example['hallucinated_answer']}
    
    --- FACTUAL PATH ANALYSIS ---
    Number of tracks: {len(factual_result['tracks'])}
    Max entropy: {factual_result['metrics']['max_entropy']:.3f}
    Entropy trajectory: {factual_result['metrics']['entropy_trajectory']}
    
    Interpretation: Entropy {'decreases monotonically' if is_decreasing(factual_result['metrics']['entropy_trajectory']) else 'shows some fluctuation'}
    → Model is {'confident' if factual_result['metrics']['max_entropy'] < 1.0 else 'uncertain'}
    
    --- HALLUCINATION PATH ANALYSIS ---
    Number of tracks: {len(halluc_result['tracks'])}
    Max entropy: {halluc_result['metrics']['max_entropy']:.3f}
    Entropy peak layer: {halluc_result['metrics']['entropy_peak_layer']}
    Birth/Death ratio: {halluc_result['metrics']['birth_death_ratio']:.3f}
    
    Critical Event Timeline:
    """
    
    # Track competition analysis
    halluc_tracks = sorted(halluc_result['tracks'], key=lambda t: max([act for _, act, _ in t.trajectory]))
    
    for i, track in enumerate(halluc_tracks[:3]):  # Top 3
        feature_label = get_feature_label(track)
        report += f"""
        
    Track {i+1} ({feature_label}):
      - Birth: Layer {track.birth_layer}
      - Death: Layer {track.death_layer if track.death_layer else 'N/A (survived)'}
      - Peak activation: {max([act for _, act, _ in track.trajectory]):.3f}
      - Final activation: {track.trajectory[-1][1]:.3f}
        """
        
    # Identify critical layer
    critical_layer = halluc_result['metrics']['entropy_peak_layer']
    report += f"""
    
    CRITICAL INSIGHT:
    At layer {critical_layer}, hypothesis entropy peaked at {halluc_result['metrics']['entropy_trajectory'][critical_layer]:.3f}.
    This indicates intense competition between semantic tracks.
    
    The winning track ('{get_feature_label(halluc_tracks[0])}') emerged victorious,
    leading to the hallucinated output: "{example['hallucinated_answer']}"
    
    Leading Indicator: MHT detected competition {critical_layer} layers deep,
    while LSD drift becomes significant at layer {critical_layer + 2} (estimated).
    """
    
    return report
PHASE 6: OPTIMIZATION & ABLATIONS (Week 6-7)
Step 6.1: Hyperparameter Tuning
File: tune_hyperparameters.py

# Grid search configuration
param_grid = {
    'sae_sparsity_lambda': [0.001, 0.01, 0.1],
    'top_k_features': [30, 50, 100],
    'association_threshold': [0.3, 0.5, 0.7],
    'birth_threshold': [0.5, 0.7, 0.9],
    'entropy_threshold': [1.0, 1.5, 2.0],
    'churn_threshold': [0.2, 0.3, 0.5]
}

# Use validation set for tuning
best_params = grid_search_cv(param_grid, val_dataset)
Step 6.2: Ablation Studies
File: ablations.py

ablations = [
    {
        'name': 'Single Hypothesis (No Competition)',
        'modification': 'Track only the strongest feature per layer',
        'purpose': 'Validate that multi-hypothesis tracking adds value'
    },
    {
        'name': 'No Track Association',
        'modification': 'Treat each layer independently, no cross-layer tracking',
        'purpose': 'Validate that temporal tracking matters'
    },
    {
        'name': 'Feature ID Association',
        'modification': 'Use original (wrong) feature ID matching instead of semantic',
        'purpose': 'Validate critique was correct about semantic similarity'
    },
    {
        'name': 'Standard ReLU SAE',
        'modification': 'Replace JumpReLU with standard ReLU SAE',
        'purpose': 'Validate that JumpReLU improves results'
    },
    {
        'name': 'Ignore Reconstruction Error',
        'modification': 'Do not use error term in tracking',
        'purpose': 'Check if hallucinations hide in reconstruction residuals'
    }
]

# Run each ablation and measure performance drop
for ablation in ablations:
    auroc_ablated = run_ablation(ablation, test_dataset)
    performance_drop = baseline_auroc - auroc_ablated
    print(f"{ablation['name']}: ΔAUROC = {performance_drop:.3f}")
PHASE 7: PAPER WRITING (Week 7-8)
Step 7.1: Key Results to Report
Main Result: AUROC on TruthfulQA (target: ≥ 0.90)
Leading Indicator: MHT detects 2-3 layers earlier than LSD
Interpretability: 20 case studies with clear track competition
Ablations: Validate all design decisions
Visualizations: Semantic radar plots + trajectory plots
Step 7.2: Paper Structure
Title: "Multi-Hypothesis Tracking for Interpretable Hallucination Detection in LLMs"

Abstract:
- Problem: Hallucinations lack mechanistic explanations
- Solution: Track competing semantic hypotheses across layers
- Results: 0.9X AUROC + interpretable explanations
- Key insight: Hallucinations emerge from track competition, not just drift

1. Introduction
   - Hallucination problem
   - Limitations of existing methods (LSD: no explanation, requires supervision)
   - Our contribution: Multi-hypothesis tracking

2. Background
   - Sparse Autoencoders
   - Mechanistic interpretability
   - LSD and related work

3. Method
   - JumpReLU SAE training
   - Track management system
   - Semantic-similarity-based association
   - Hypothesis entropy metric

4. Experiments
   - TruthfulQA evaluation
   - Comparison with LSD, semantic entropy
   - Leading indicator analysis

5. Case Studies
   - 20 detailed examples
   - Semantic radar visualizations
   - Track competition narratives

6. Ablations & Analysis
   - Validate design choices
   - Error analysis

7. Discussion
   - Interpretability advantages
   - Limitations
   - Future work

8. Conclusion
Step 7.3: Code Release
Repository Structure:

mht-hallucination/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   └── download_truthfulqa.py
├── models/
│   ├── sae_model.py
│   ├── model_wrapper.py
│   └── checkpoints/
├── tracking/
│   ├── feature_extractor.py
│   ├── hypothesis_tracker.py
│   └── track_association.py
├── detection/
│   ├── detector.py
│   └── divergence_metrics.py
├── evaluation/
│   ├── evaluate.py
│   └── ablations.py
├── visualization/
│   ├── visualize.py
│   └── case_studies.py
├── notebooks/
│   └── demo.ipynb
└── scripts/
    ├── train_saes.sh
    └── run_evaluation.sh
EXECUTION TIMELINE (CRITICAL PATH)
Week 1: Foundation ✓
[ ] Day 1-2: Data pipeline + model wrapper
[ ] Day 3-4: Verify residual stream extraction works
[ ] Day 5-7: Begin SAE implementation
Week 2-3: SAE Training ✓
[ ] Day 8-14: Implement JumpReLU SAE
[ ] Day 15-21: Train 12 SAEs (parallelizable)
[ ] Validate reconstruction quality
Week 3-4: Tracking System ✓
[ ] Day 22-25: Feature extractor
[ ] Day 26-28: Track manager + semantic association
[ ] Day 29-30: Test on 10 examples
Week 4-5: Detection Pipeline ✓
[ ] Day 31-35: Implement all metrics (especially entropy)
[ ] Day 36-37: Build detector
[ ] Day 38-40: Evaluate on TruthfulQA
[ ] Milestone: Achieve AUROC ≥ 0.90
Week 5-6: Visualization ✓
[ ] Day 41-43: Semantic radar plots
[ ] Day 44-45: Trajectory plots
[ ] Day 46-49: Generate 20 case studies
Week 6-7: Optimization ✓
[ ] Day 50-53: Hyperparameter tuning
[ ] Day 54-56: Run all ablations
Week 7-8: Writing ✓
[ ] Day 57-60: Paper draft
[ ] Day 61-63: Code cleanup + documentation
[ ] Day 64: Final review
SUCCESS CRITERIA
Minimum Viable (Conference Acceptance)
✅ AUROC ≥ 0.90 on TruthfulQA
✅ 10 clear case studies showing track competition
✅ Semantic radar visualization working
✅ At least 3 successful ablations
Strong Contribution (Spotlight/Oral)
✅ AUROC ≥ 0.92 (beat LSD's 0.96 is unrealistic, match it)
✅ 20 case studies with diverse error types
✅ Leading indicator validated (2-3 layer advantage)
✅ All 5 ablations successful
✅ Code + pretrained SAEs released
Exceptional (Best Paper Contender)
✅ AUROC > 0.94
✅ Works on multiple models (GPT-2, GPT-2-medium, GPT-2-large)
✅ Validated on multiple datasets (TruthfulQA + HaluEval)
✅ Theoretical analysis of why entropy predicts hallucinations
RISK MITIGATION
RiskProbabilityImpactMitigationSAEs don't learn interpretable featuresMediumHighUse JumpReLU, increase sparsity, try different architecturesSemantic association failsLowCriticalValidate on synthetic data first, tune cost functionAUROC < 0.90MediumHighAdd ensemble methods, try different classifiersNo clear track competition in case studiesLowMediumSelect different question types, focus on factual recallCan't beat LSD performanceHighMediumEmphasize interpretability, not raw performanceFINAL CHECKLIST
Before Starting
[ ] Confirm GPU access (A100 or equivalent)
[ ] Download TruthfulQA dataset
[ ] Install all dependencies
[ ] Verify GPT-2 loads correctly
Critical Validations
[ ] JumpReLU SAE reconstruction loss < 0.01
[ ] Feature interpretation makes semantic sense
[ ] Track association produces stable trajectories
[ ] Entropy metric shows expected patterns
[ ] Leading indicator analysis shows 2-3 layer advantage
Paper Submission
[ ] All figures publication-ready
[ ] Code repository public + documented
[ ] Pretrained SAEs available for download
[ ] Reproducibility verified on clean machine
CONCLUSION
This design document represents a feasible, high-impact research project that:

