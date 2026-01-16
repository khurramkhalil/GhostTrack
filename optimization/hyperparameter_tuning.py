"""
Hyperparameter tuning for GhostTrack.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import itertools
from tqdm import tqdm

from tracking import HypothesisTracker
from detection import HallucinationDetector


class GridSearchCV:
    """Grid search with cross-validation for hyperparameter tuning."""

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'roc_auc',
        verbose: bool = True
    ):
        """
        Initialize grid search.

        Args:
            param_grid: Dictionary of parameter names to lists of values.
            cv: Number of cross-validation folds.
            scoring: Scoring metric.
            verbose: Whether to show progress.
        """
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray,
        model_type: str = 'random_forest',
        num_layers: int = 12
    ) -> 'GridSearchCV':
        """
        Fit grid search.

        Args:
            trackers: List of HypothesisTracker instances.
            labels: Binary labels.
            model_type: Type of detector model.
            num_layers: Number of layers.

        Returns:
            Self.
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))

        best_score = -np.inf

        iterator = tqdm(param_combinations) if self.verbose else param_combinations

        for params in iterator:
            param_dict = dict(zip(param_names, params))

            if self.verbose:
                iterator.set_description(f"Testing {param_dict}")

            # Train detector with these parameters
            detector = HallucinationDetector(
                model_type=model_type,
                num_layers=num_layers,
                **param_dict
            )

            # Perform cross-validation
            scores = []
            fold_size = len(trackers) // self.cv

            for fold in range(self.cv):
                # Split data
                val_start = fold * fold_size
                val_end = val_start + fold_size

                train_trackers = trackers[:val_start] + trackers[val_end:]
                train_labels = np.concatenate([labels[:val_start], labels[val_end:]])
                val_trackers = trackers[val_start:val_end]
                val_labels = labels[val_start:val_end]

                # Train
                detector.fit(train_trackers, train_labels)

                # Evaluate
                predictions = detector.predict_proba(val_trackers)[:, 1]
                score = roc_auc_score(val_labels, predictions)
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.results_.append({
                'params': param_dict,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            })

            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = param_dict
                self.best_score_ = mean_score

                if self.verbose:
                    print(f"\nNew best score: {best_score:.4f} with params {param_dict}")

        return self

    def get_results_dataframe(self):
        """Get results as pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.results_)
        except ImportError:
            return self.results_


class HyperparameterTuner:
    """Hyperparameter tuning for tracking and detection."""

    def __init__(self, num_layers: int = 12):
        """
        Initialize tuner.

        Args:
            num_layers: Number of layers.
        """
        self.num_layers = num_layers
        self.results = {}

    def tune_tracking_params(
        self,
        trackers_grid: Dict[str, List[HypothesisTracker]],
        labels: np.ndarray,
        model_type: str = 'random_forest'
    ) -> Dict:
        """
        Tune tracking hyperparameters.

        Evaluates different tracking configurations by training detectors
        on trackers generated with different parameters.

        Args:
            trackers_grid: Dict mapping config names to lists of trackers.
            labels: Binary labels.
            model_type: Detector model type.

        Returns:
            Dictionary with results.
        """
        results = {}

        for config_name, trackers in trackers_grid.items():
            print(f"\nEvaluating tracking config: {config_name}")

            # Train detector
            detector = HallucinationDetector(
                model_type=model_type,
                num_layers=self.num_layers
            )

            # Cross-validation
            fold_size = len(trackers) // 5
            scores = []

            for fold in range(5):
                val_start = fold * fold_size
                val_end = val_start + fold_size

                train_trackers = trackers[:val_start] + trackers[val_end:]
                train_labels = np.concatenate([labels[:val_start], labels[val_end:]])
                val_trackers = trackers[val_start:val_end]
                val_labels = labels[val_start:val_end]

                detector.fit(train_trackers, train_labels)
                predictions = detector.predict_proba(val_trackers)[:, 1]
                score = roc_auc_score(val_labels, predictions)
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            results[config_name] = {
                'mean_auroc': mean_score,
                'std_auroc': std_score,
                'scores': scores
            }

            print(f"  AUROC: {mean_score:.4f} ± {std_score:.4f}")

        # Find best config
        best_config = max(results.items(), key=lambda x: x[1]['mean_auroc'])

        self.results['tracking'] = {
            'all_results': results,
            'best_config': best_config[0],
            'best_score': best_config[1]['mean_auroc']
        }

        return results

    def tune_detector_params(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray
    ) -> Dict:
        """
        Tune detector hyperparameters.

        Args:
            trackers: List of trackers.
            labels: Binary labels.

        Returns:
            Dictionary with results for each model type.
        """
        model_types = ['random_forest', 'gradient_boosting', 'logistic', 'svm']
        results = {}

        for model_type in model_types:
            print(f"\nTuning {model_type}...")

            if model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif model_type == 'logistic':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear']
                }
            else:
                continue

            grid_search = GridSearchCV(
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                verbose=True
            )

            grid_search.fit(trackers, labels, model_type=model_type, num_layers=self.num_layers)

            results[model_type] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'all_results': grid_search.results_
            }

            print(f"  Best score: {grid_search.best_score_:.4f}")
            print(f"  Best params: {grid_search.best_params_}")

        # Find overall best
        best_model = max(results.items(), key=lambda x: x[1]['best_score'])

        self.results['detector'] = {
            'all_results': results,
            'best_model': best_model[0],
            'best_params': best_model[1]['best_params'],
            'best_score': best_model[1]['best_score']
        }

        return results

    def generate_tuning_report(self, output_path: str = './optimization/tuning_report.md'):
        """Generate markdown report of tuning results."""
        report = """# Hyperparameter Tuning Report

## Overview

This report summarizes the results of hyperparameter tuning for GhostTrack.

---

"""
        # Tracking results
        if 'tracking' in self.results:
            report += """## Tracking Configuration Tuning

"""
            tracking_results = self.results['tracking']['all_results']

            report += "| Configuration | Mean AUROC | Std AUROC |\n"
            report += "|---------------|------------|----------|\n"

            for config, metrics in tracking_results.items():
                report += f"| {config} | {metrics['mean_auroc']:.4f} | {metrics['std_auroc']:.4f} |\n"

            report += f"""
**Best Configuration**: {self.results['tracking']['best_config']}
**Best Score**: {self.results['tracking']['best_score']:.4f}

---

"""

        # Detector results
        if 'detector' in self.results:
            report += """## Detector Model Tuning

"""
            detector_results = self.results['detector']['all_results']

            report += "| Model Type | Best AUROC | Best Parameters |\n"
            report += "|------------|------------|----------------|\n"

            for model, metrics in detector_results.items():
                params_str = ', '.join([f"{k}={v}" for k, v in metrics['best_params'].items()])
                report += f"| {model} | {metrics['best_score']:.4f} | {params_str} |\n"

            report += f"""
**Best Model**: {self.results['detector']['best_model']}
**Best Score**: {self.results['detector']['best_score']:.4f}
**Best Parameters**: {self.results['detector']['best_params']}

---

## Recommendations

Based on the tuning results:

1. Use **{self.results['detector']['best_model']}** as the detector model
2. Configure with parameters: {self.results['detector']['best_params']}
3. Expected AUROC: {self.results['detector']['best_score']:.4f}

"""

        # Save report
        from pathlib import Path
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            f.write(report)

        print(f"\nTuning report saved to {output_path}")

        return report
