"""
Track trajectory visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from tracking import HypothesisTracker, Track


def plot_track_trajectories(
    tracker: HypothesisTracker,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    show_top_k: int = 10
) -> plt.Figure:
    """
    Plot track trajectories showing activation evolution across layers.

    Args:
        tracker: HypothesisTracker instance with completed tracking.
        save_path: Optional path to save figure.
        figsize: Figure size.
        show_top_k: Number of top tracks to show.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Track Trajectories Across Layers', fontsize=16, fontweight='bold')

    # Get top tracks by max activation
    all_tracks = tracker.get_all_tracks()
    if not all_tracks:
        return fig

    # Sort by max activation
    sorted_tracks = sorted(
        all_tracks,
        key=lambda t: max([obs.activation for obs in t.observations]),
        reverse=True
    )[:show_top_k]

    # Plot 1: Activation trajectories
    ax1 = axes[0, 0]
    for track in sorted_tracks:
        layers = [obs.layer_idx for obs in track.observations]
        activations = [obs.activation for obs in track.observations]
        ax1.plot(layers, activations, marker='o', alpha=0.6, linewidth=2)

    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Activation', fontsize=12)
    ax1.set_title(f'Top {show_top_k} Track Activations', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Track lifespans
    ax2 = axes[0, 1]
    for i, track in enumerate(sorted_tracks):
        layers = [obs.layer_idx for obs in track.observations]
        ax2.barh(i, len(layers), left=min(layers), alpha=0.6)

    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Track ID', fontsize=12)
    ax2.set_title('Track Lifespans', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(sorted_tracks)))
    ax2.set_yticklabels([f'Track {i}' for i in range(len(sorted_tracks))])
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Activation distribution
    ax3 = axes[1, 0]
    all_activations = []
    for track in all_tracks:
        all_activations.extend([obs.activation for obs in track.observations])

    if all_activations:
        ax3.hist(all_activations, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Activation', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('Activation Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Track count per layer
    ax4 = axes[1, 1]
    layer_counts = {}
    for track in all_tracks:
        for obs in track.observations:
            layer_counts[obs.layer_idx] = layer_counts.get(obs.layer_idx, 0) + 1

    if layer_counts:
        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]
        ax4.bar(layers, counts, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Layer Index', fontsize=12)
        ax4.set_ylabel('Number of Tracks', fontsize=12)
        ax4.set_title('Track Count per Layer', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")

    return fig


def plot_competition_heatmap(
    tracker: HypothesisTracker,
    num_layers: int = 12,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap showing competition between tracks across layers.

    Args:
        tracker: HypothesisTracker instance.
        num_layers: Total number of layers.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    # Build competition matrix: tracks x layers
    all_tracks = tracker.get_all_tracks()
    if not all_tracks:
        return plt.figure()

    # Sort tracks by birth layer
    sorted_tracks = sorted(all_tracks, key=lambda t: t.birth_layer)

    # Create activation matrix
    num_tracks = len(sorted_tracks)
    activation_matrix = np.zeros((num_tracks, num_layers))

    for i, track in enumerate(sorted_tracks):
        for obs in track.observations:
            if obs.layer_idx < num_layers:
                activation_matrix[i, obs.layer_idx] = obs.activation

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        activation_matrix,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Activation'},
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Track ID', fontsize=12)
    ax.set_title('Hypothesis Competition Heatmap', fontsize=14, fontweight='bold')

    # Add track birth/death markers
    for i, track in enumerate(sorted_tracks):
        # Mark birth
        ax.scatter(track.birth_layer + 0.5, i + 0.5, marker='>',
                  color='lime', s=100, edgecolors='black', linewidths=1.5,
                  label='Birth' if i == 0 else '')

        # Mark death
        if track.death_layer is not None and track.death_layer < num_layers:
            ax.scatter(track.death_layer + 0.5, i + 0.5, marker='x',
                      color='red', s=100, linewidths=2,
                      label='Death' if i == 0 else '')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], loc='upper right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved competition heatmap to {save_path}")

    return fig


def plot_divergence_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot divergence metrics in organized panels.

    Args:
        metrics: Dictionary of divergence metrics.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    # Group metrics by family
    families = {
        'Entropy': [k for k in metrics.keys() if k.startswith('entropy_')],
        'Churn': [k for k in metrics.keys() if k.startswith('churn_')],
        'Competition': [k for k in metrics.keys() if k.startswith('competition_')],
        'Stability': [k for k in metrics.keys() if k.startswith('stability_')],
        'Dominance': [k for k in metrics.keys() if k.startswith('dominance_')],
        'Density': [k for k in metrics.keys() if k.startswith('density_')]
    }

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Divergence Metrics Overview', fontsize=16, fontweight='bold')

    for idx, (family_name, metric_keys) in enumerate(families.items()):
        if not metric_keys:
            continue

        ax = axes[idx // 2, idx % 2]

        # Extract values
        values = [metrics[k] for k in metric_keys]
        labels = [k.replace(f'{family_name.lower()}_', '') for k in metric_keys]

        # Create bar plot
        bars = ax.barh(labels, values, alpha=0.7, edgecolor='black')

        # Color bars by value
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        colors = plt.cm.viridis(norm(values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Value', fontsize=10)
        ax.set_title(f'{family_name} Metrics', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved divergence metrics plot to {save_path}")

    return fig


def plot_activation_timeline(
    tracker: HypothesisTracker,
    num_layers: int = 12,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot activation timeline showing births and deaths across layers.

    Args:
        tracker: HypothesisTracker instance.
        num_layers: Total number of layers.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Track Birth/Death Timeline', fontsize=16, fontweight='bold')

    all_tracks = tracker.get_all_tracks()
    if not all_tracks:
        return fig

    # Count births and deaths per layer
    births_per_layer = np.zeros(num_layers)
    deaths_per_layer = np.zeros(num_layers)

    for track in all_tracks:
        if track.birth_layer < num_layers:
            births_per_layer[track.birth_layer] += 1
        if track.death_layer is not None and track.death_layer < num_layers:
            deaths_per_layer[track.death_layer] += 1

    # Plot births
    ax1 = axes[0]
    layers = np.arange(num_layers)
    ax1.bar(layers, births_per_layer, alpha=0.7, color='green', edgecolor='black')
    ax1.set_ylabel('Birth Count', fontsize=12)
    ax1.set_title('Track Births per Layer', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot deaths
    ax2 = axes[1]
    ax2.bar(layers, deaths_per_layer, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Death Count', fontsize=12)
    ax2.set_title('Track Deaths per Layer', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved activation timeline to {save_path}")

    return fig
