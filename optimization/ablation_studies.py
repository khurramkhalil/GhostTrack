"""
Ablation studies for feature importance analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

from tracking import HypothesisTracker
from detection import HallucinationDetector, DivergenceMetrics


class FeatureSelector:
    """Feature selection using importance and ablation."""

    def __init__(self, num_layers: int = 12):
        """
        Initialize feature selector.

        Args:
            num_layers: Number of layers.
        """
        self.num_layers = num_layers
        self.feature_importance_ = None
        self.selected_features_ = None

    def compute_importance(
        self,
        detector: HallucinationDetector
    ) -> np.ndarray:
        """
        Get feature importance from trained detector.

        Args:
            detector: Trained detector.

        Returns:
            Array of feature importances.
        """
        importance = detector.get_feature_importance()
        self.feature_importance_ = importance
        return importance

    def select_top_k(self, k: int = 10) -> List[int]:
        """
        Select top k most important features.

        Args:
            k: Number of features to select.

        Returns:
            List of feature indices.
        """
        if self.feature_importance_ is None:
            raise ValueError("Must compute importance first")

        top_indices = np.argsort(self.feature_importance_)[-k:][::-1]
        self.selected_features_ = top_indices.tolist()

        return self.selected_features_

    def select_by_threshold(self, threshold: float = 0.01) -> List[int]:
        """
        Select features above importance threshold.

        Args:
            threshold: Minimum importance.

        Returns:
            List of feature indices.
        """
        if self.feature_importance_ is None:
            raise ValueError("Must compute importance first")

        selected = np.where(self.feature_importance_ > threshold)[0]
        self.selected_features_ = selected.tolist()

        return self.selected_features_


class AblationStudy:
    """Conduct ablation studies to understand feature contributions."""

    def __init__(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray,
        num_layers: int = 12
    ):
        """
        Initialize ablation study.

        Args:
            trackers: List of HypothesisTracker instances.
            labels: Binary labels.
            num_layers: Number of layers.
        """
        self.trackers = trackers
        self.labels = labels
        self.num_layers = num_layers
        self.results = {}

    def ablate_metric_families(
        self,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Ablate metric families to measure their contribution.

        Tests performance when removing each family of metrics.

        Args:
            model_type: Detector model type.

        Returns:
            Dictionary with ablation results.
        """
        # Get all metric names
        sample_metrics = DivergenceMetrics.compute_all_metrics(
            self.trackers[0], self.num_layers
        )
        all_features = list(sample_metrics.keys())

        # Define metric families
        families = {
            'entropy': [f for f in all_features if f.startswith('entropy_')],
            'churn': [f for f in all_features if f.startswith('churn_')],
            'competition': [f for f in all_features if f.startswith('competition_')],
            'stability': [f for f in all_features if f.startswith('stability_')],
            'dominance': [f for f in all_features if f.startswith('dominance_')],
            'density': [f for f in all_features if f.startswith('density_')]
        }

        results = {}

        # Baseline: all features
        print("Training baseline with all features...")
        baseline_metrics = self._train_and_evaluate(
            self.trackers, self.labels, model_type, features_to_exclude=[]
        )
        results['baseline'] = baseline_metrics

        print(f"Baseline AUROC: {baseline_metrics['auroc']:.4f}")

        # Ablate each family
        for family_name, family_features in families.items():
            print(f"\nAblating {family_name} family ({len(family_features)} features)...")

            metrics = self._train_and_evaluate(
                self.trackers, self.labels, model_type,
                features_to_exclude=family_features
            )

            drop = baseline_metrics['auroc'] - metrics['auroc']
            results[f'ablate_{family_name}'] = {
                **metrics,
                'auroc_drop': drop
            }

            print(f"  AUROC: {metrics['auroc']:.4f} (drop: {drop:+.4f})")

        self.results['family_ablation'] = results

        return results

    def ablate_individual_features(
        self,
        model_type: str = 'random_forest',
        top_k: int = 10
    ) -> Dict:
        """
        Ablate individual features to measure their contribution.

        Args:
            model_type: Detector model type.
            top_k: Number of top features to test.

        Returns:
            Dictionary with ablation results.
        """
        # Train baseline
        detector = HallucinationDetector(
            model_type=model_type,
            num_layers=self.num_layers
        )

        # Split data
        split = int(0.8 * len(self.trackers))
        train_trackers = self.trackers[:split]
        train_labels = self.labels[:split]
        test_trackers = self.trackers[split:]
        test_labels = self.labels[split:]

        detector.fit(train_trackers, train_labels)

        # Get feature importance
        importance = detector.get_feature_importance()
        if importance is None:
            print("Model doesn't support feature importance")
            return {}

        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]

        # Get feature names
        sample_metrics = DivergenceMetrics.compute_all_metrics(
            self.trackers[0], self.num_layers
        )
        feature_names = list(sample_metrics.keys())

        # Baseline performance
        baseline_metrics = detector.evaluate(test_trackers, test_labels)

        results = {'baseline': baseline_metrics}

        # Ablate each top feature
        for idx in tqdm(top_indices, desc="Ablating features"):
            feature_name = feature_names[idx]

            metrics = self._train_and_evaluate(
                self.trackers, self.labels, model_type,
                features_to_exclude=[feature_name]
            )

            drop = baseline_metrics['auroc'] - metrics['auroc']
            results[feature_name] = {
                **metrics,
                'auroc_drop': drop,
                'importance': float(importance[idx])
            }

        self.results['individual_ablation'] = results

        return results

    def cumulative_feature_addition(
        self,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Add features cumulatively in order of importance.

        Args:
            model_type: Detector model type.

        Returns:
            Dictionary with results.
        """
        # Train full model to get importance
        detector = HallucinationDetector(
            model_type=model_type,
            num_layers=self.num_layers
        )

        split = int(0.8 * len(self.trackers))
        train_trackers = self.trackers[:split]
        train_labels = self.labels[:split]

        detector.fit(train_trackers, train_labels)

        # Get feature importance
        importance = detector.get_feature_importance()
        if importance is None:
            return {}

        # Sort features by importance
        sorted_indices = np.argsort(importance)[::-1]

        # Get feature names
        sample_metrics = DivergenceMetrics.compute_all_metrics(
            self.trackers[0], self.num_layers
        )
        feature_names = list(sample_metrics.keys())

        results = {}

        # Test with increasing number of features
        for num_features in [1, 2, 5, 10, 15, 20, len(feature_names)]:
            if num_features > len(sorted_indices):
                break

            features_to_keep = [feature_names[i] for i in sorted_indices[:num_features]]
            features_to_exclude = [f for f in feature_names if f not in features_to_keep]

            metrics = self._train_and_evaluate(
                self.trackers, self.labels, model_type,
                features_to_exclude=features_to_exclude
            )

            results[f'top_{num_features}'] = metrics

            print(f"Top {num_features} features: AUROC = {metrics['auroc']:.4f}")

        self.results['cumulative_addition'] = results

        return results

    def _train_and_evaluate(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray,
        model_type: str,
        features_to_exclude: List[str]
    ) -> Dict:
        """
        Train and evaluate with specific features excluded.

        Args:
            trackers: List of trackers.
            labels: Binary labels.
            model_type: Model type.
            features_to_exclude: List of feature names to exclude.

        Returns:
            Evaluation metrics.
        """
        # Extract features manually and exclude specific ones
        from detection import DivergenceMetrics

        X = []
        for tracker in trackers:
            metrics = DivergenceMetrics.compute_all_metrics(tracker, self.num_layers)

            # Remove excluded features
            for feature in features_to_exclude:
                metrics.pop(feature, None)

            X.append(list(metrics.values()))

        X = np.array(X)

        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = labels[:split], labels[split:]

        # Train model directly on features
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return {
            'auroc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

    def generate_ablation_report(
        self,
        output_path: str = './optimization/ablation_report.md'
    ) -> str:
        """
        Generate ablation study report.

        Args:
            output_path: Path to save report.

        Returns:
            Report string.
        """
        report = """# Ablation Study Report

## Overview

This report summarizes ablation studies conducted on GhostTrack features.

---

"""
        # Family ablation
        if 'family_ablation' in self.results:
            report += """## Metric Family Ablation

Shows the impact of removing each family of metrics.

| Family | AUROC | AUROC Drop | Impact |
|--------|-------|------------|--------|
"""
            baseline_auroc = self.results['family_ablation']['baseline']['auroc']
            report += f"| Baseline (all) | {baseline_auroc:.4f} | - | - |\n"

            for key, metrics in self.results['family_ablation'].items():
                if key == 'baseline':
                    continue

                family_name = key.replace('ablate_', '')
                auroc = metrics['auroc']
                drop = metrics['auroc_drop']
                impact = 'High' if drop > 0.05 else ('Medium' if drop > 0.02 else 'Low')

                report += f"| Without {family_name} | {auroc:.4f} | {drop:+.4f} | {impact} |\n"

            report += "\n---\n\n"

        # Individual ablation
        if 'individual_ablation' in self.results:
            report += """## Individual Feature Ablation

Top features by importance, showing impact when removed.

| Feature | Importance | AUROC Drop | Impact |
|---------|-----------|------------|--------|
"""
            results = self.results['individual_ablation']
            baseline_auroc = results['baseline']['auroc']

            # Sort by AUROC drop
            features = [(k, v) for k, v in results.items() if k != 'baseline']
            features.sort(key=lambda x: x[1]['auroc_drop'], reverse=True)

            for feature_name, metrics in features[:15]:  # Top 15
                importance = metrics['importance']
                drop = metrics['auroc_drop']
                impact = 'Critical' if drop > 0.10 else ('High' if drop > 0.05 else 'Medium')

                report += f"| {feature_name} | {importance:.4f} | {drop:+.4f} | {impact} |\n"

            report += "\n---\n\n"

        # Cumulative addition
        if 'cumulative_addition' in self.results:
            report += """## Cumulative Feature Addition

Performance with increasing numbers of top features.

| Num Features | AUROC | Accuracy | F1 |
|--------------|-------|----------|-----|
"""
            for key, metrics in sorted(self.results['cumulative_addition'].items()):
                num = key.replace('top_', '')
                report += f"| {num} | {metrics['auroc']:.4f} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} |\n"

            report += "\n---\n\n"

        report += """## Conclusions

1. **Most Important Metric Families**: Based on ablation, identify which families contribute most
2. **Critical Features**: Individual features with highest impact
3. **Minimum Feature Set**: Smallest set achieving good performance

---

*Generated by GhostTrack Ablation Study*
"""

        # Save report
        from pathlib import Path
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            f.write(report)

        print(f"\nAblation report saved to {output_path}")

        return report
