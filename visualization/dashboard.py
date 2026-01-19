"""
Interactive dashboard for GhostTrack visualization.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

from tracking import HypothesisTracker
from detection import DivergenceMetrics


def create_interactive_dashboard(
    tracker: HypothesisTracker,
    text: str,
    prediction: float,
    is_hallucination: bool,
    num_layers: int = 12,
    output_dir: str = './dashboard'
) -> Dict:
    """
    Create interactive dashboard data for a single example.

    Args:
        tracker: HypothesisTracker instance.
        text: Input text.
        prediction: Model prediction (0-1).
        is_hallucination: Ground truth label.
        num_layers: Number of layers.
        output_dir: Output directory for dashboard files.

    Returns:
        Dictionary with dashboard data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute divergence metrics
    metrics_computer = DivergenceMetrics()
    metrics = metrics_computer.compute_all_metrics(tracker, num_layers)

    # Get track statistics
    all_tracks = tracker.tracks
    track_stats = {
        'total_tracks': len(all_tracks),
        'active_tracks': len([t for t in all_tracks if t.death_layer is None]),
        'dead_tracks': len([t for t in all_tracks if t.death_layer is not None]),
        'avg_lifespan': np.mean([len(t.trajectory) for t in all_tracks]) if all_tracks else 0,
        'max_activation': max([max([act for _, act, _ in t.trajectory])
                              for t in all_tracks]) if all_tracks else 0
    }

    # Build track timeline data
    track_timeline = []
    for i, track in enumerate(all_tracks):
        track_data = {
            'track_id': i,
            'birth_layer': track.birth_layer,
            'death_layer': track.death_layer,
            'observations': [
                {
                    'layer': layer,
                    'activation': float(activation),
                    'feature_id': -1,  # Feature ID not strictly tracked
                    'token_pos': track.token_pos
                }
                for layer, activation, _ in track.trajectory
            ]
        }
        track_timeline.append(track_data)

    # Build layer-by-layer statistics
    layer_stats = []
    for layer_idx in range(num_layers):
        layer_tracks = [
            t for t in all_tracks
            if any(layer == layer_idx for layer, _, _ in t.trajectory)
        ]

        births = len([t for t in all_tracks if t.birth_layer == layer_idx])
        deaths = len([t for t in all_tracks if t.death_layer == layer_idx])

        activations = []
        for track in layer_tracks:
            for layer, activation, _ in track.trajectory:
                if layer == layer_idx:
                    activations.append(activation)

        layer_stats.append({
            'layer': layer_idx,
            'num_tracks': len(layer_tracks),
            'births': births,
            'deaths': deaths,
            'mean_activation': float(np.mean(activations)) if activations else 0.0,
            'max_activation': float(np.max(activations)) if activations else 0.0
        })

    # Compile dashboard data
    dashboard_data = {
        'metadata': {
            'text': text,
            'text_length': len(text),
            'prediction': float(prediction),
            'is_hallucination': is_hallucination,
            'num_layers': num_layers
        },
        'track_stats': track_stats,
        'divergence_metrics': {k: float(v) for k, v in metrics.items()},
        'track_timeline': track_timeline,
        'layer_stats': layer_stats
    }

    # Save to JSON
    json_path = output_path / 'dashboard_data.json'
    with open(json_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data saved to {json_path}")

    # Generate HTML dashboard
    html_content = generate_html_dashboard(dashboard_data)
    html_path = output_path / 'dashboard.html'
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Interactive dashboard saved to {html_path}")

    return dashboard_data


def generate_html_dashboard(data: Dict) -> str:
    """
    Generate HTML for interactive dashboard.

    Args:
        data: Dashboard data dictionary.

    Returns:
        HTML string.
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>GhostTrack Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .prediction {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            margin-top: 10px;
            font-weight: bold;
        }}
        .prediction.hallucination {{
            background-color: #ff4444;
        }}
        .prediction.factual {{
            background-color: #44ff44;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .stat:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
        }}
        .stat-value {{
            color: #333;
            font-weight: bold;
        }}
        .plot {{
            width: 100%;
            height: 400px;
        }}
        .text-preview {{
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            font-family: monospace;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 GhostTrack Dashboard</h1>
            <div class="prediction {'hallucination' if data['metadata']['is_hallucination'] else 'factual'}">
                Prediction: {data['metadata']['prediction']:.3f} |
                Ground Truth: {'HALLUCINATION' if data['metadata']['is_hallucination'] else 'FACTUAL'}
            </div>
        </div>

        <div class="card">
            <h2>📝 Input Text</h2>
            <div class="text-preview">{data['metadata']['text']}</div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Track Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Total Tracks:</span>
                    <span class="stat-value">{data['track_stats']['total_tracks']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Active Tracks:</span>
                    <span class="stat-value">{data['track_stats']['active_tracks']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Dead Tracks:</span>
                    <span class="stat-value">{data['track_stats']['dead_tracks']}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Lifespan:</span>
                    <span class="stat-value">{data['track_stats']['avg_lifespan']:.2f} layers</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Max Activation:</span>
                    <span class="stat-value">{data['track_stats']['max_activation']:.3f}</span>
                </div>
            </div>

            <div class="card">
                <h2>🎯 Key Metrics</h2>
                <div class="stat">
                    <span class="stat-label">Entropy (Mean):</span>
                    <span class="stat-value">{data['divergence_metrics'].get('entropy_mean', 0):.3f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Churn Rate:</span>
                    <span class="stat-value">{data['divergence_metrics'].get('churn_rate', 0):.3f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Competition:</span>
                    <span class="stat-value">{data['divergence_metrics'].get('competition_mean', 0):.3f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Stability:</span>
                    <span class="stat-value">{data['divergence_metrics'].get('stability_mean', 0):.3f}</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📈 Track Trajectories</h2>
            <div id="trajectoryPlot" class="plot"></div>
        </div>

        <div class="card">
            <h2>🔥 Layer Activity</h2>
            <div id="layerPlot" class="plot"></div>
        </div>
    </div>

    <script>
        // Track trajectories plot
        const trackData = {json.dumps([
            {
                'x': [obs['layer'] for obs in track['observations']],
                'y': [obs['activation'] for obs in track['observations']],
                'mode': 'lines+markers',
                'name': f"Track {track['track_id']}",
                'line': {'width': 2},
                'marker': {'size': 6}
            }
            for track in data['track_timeline'][:10]
        ])};

        const trajectoryLayout = {{
            title: 'Track Activation Trajectories (Top 10)',

            xaxis: {{title: 'Layer Index'}},
            yaxis: {{title: 'Activation'}},
            hovermode: 'closest'
        }};

        Plotly.newPlot('trajectoryPlot', trackData, trajectoryLayout);

        // Layer activity plot
        const layerData = [
            {{
                x: {json.dumps([s['layer'] for s in data['layer_stats']])},
                y: {json.dumps([s['num_tracks'] for s in data['layer_stats']])},
                type: 'bar',
                name: 'Active Tracks',
                marker: {{color: '#667eea'}}
            }},
            {{
                x: {json.dumps([s['layer'] for s in data['layer_stats']])},
                y: {json.dumps([s['births'] for s in data['layer_stats']])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Births',
                marker: {{color: 'green', size: 8}},
                line: {{width: 3}}
            }},
            {{
                x: {json.dumps([s['layer'] for s in data['layer_stats']])},
                y: {json.dumps([s['deaths'] for s in data['layer_stats']])},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Deaths',
                marker: {{color: 'red', size: 8}},
                line: {{width: 3}}
            }}
        ];

        const layerLayout = {{
            title: 'Layer-wise Track Activity',
            xaxis: {{title: 'Layer Index'}},
            yaxis: {{title: 'Count'}},
            barmode: 'group'
        }};

        Plotly.newPlot('layerPlot', layerData, layerLayout);
    </script>
</body>
</html>
"""
    return html
