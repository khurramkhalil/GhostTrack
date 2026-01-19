"""
Hallucination detector using track divergence metrics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import pickle
from pathlib import Path

from tracking.hypothesis_tracker import HypothesisTracker
from .divergence_metrics import DivergenceMetrics


class HallucinationDetector:
    """
    Binary classifier for hallucination detection.

    Uses divergence metrics from hypothesis tracking to classify
    text as factual (0) or hallucinated (1).
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        num_layers: int = 12,
        **model_kwargs
    ):
        """
        Initialize detector.

        Args:
            model_type: Type of classifier ('random_forest', 'gradient_boosting',
                       'logistic_regression', 'svm', or 'ensemble').
            num_layers: Number of transformer layers.
            **model_kwargs: Additional arguments for the classifier.
        """
        self.model_type = model_type
        self.num_layers = num_layers
        self.metrics_computer = DivergenceMetrics()
        self.scaler = StandardScaler()

        # Initialize classifier
        if model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', None),
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=model_kwargs.get('n_estimators', 100),
                learning_rate=model_kwargs.get('learning_rate', 0.1),
                max_depth=model_kwargs.get('max_depth', 3),
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                C=model_kwargs.get('C', 1.0),
                max_iter=model_kwargs.get('max_iter', 1000),
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'svm':
            self.classifier = SVC(
                C=model_kwargs.get('C', 1.0),
                kernel=model_kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'ensemble':
            # Ensemble of multiple classifiers
            self.classifiers = [
                RandomForestClassifier(n_estimators=100, random_state=42),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                LogisticRegression(max_iter=1000, random_state=42)
            ]
            self.scaler = [StandardScaler() for _ in self.classifiers]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.is_fitted = False

    def extract_features_from_tracker(
        self,
        tracker: HypothesisTracker
    ) -> np.ndarray:
        """
        Extract feature vector from tracker.

        Args:
            tracker: HypothesisTracker instance.

        Returns:
            Feature vector as numpy array.
        """
        return self.metrics_computer.get_feature_vector(tracker, self.num_layers)

    def extract_features_from_trackers(
        self,
        trackers: List[HypothesisTracker]
    ) -> np.ndarray:
        """
        Extract features from multiple trackers.

        Args:
            trackers: List of HypothesisTracker instances.

        Returns:
            Feature matrix [n_samples, n_features].
        """
        features = []
        for tracker in trackers:
            feat_vec = self.extract_features_from_tracker(tracker)
            features.append(feat_vec)

        return np.array(features)

    def fit(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray
    ):
        """
        Train the detector.

        Args:
            trackers: List of HypothesisTracker instances.
            labels: Binary labels (0=factual, 1=hallucinated).
        """
        # Extract features
        X = self.extract_features_from_trackers(trackers)

        if self.model_type == 'ensemble':
            # Train each classifier in ensemble
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].fit_transform(X)
                clf.fit(X_scaled, labels)
        else:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train classifier
            self.classifier.fit(X_scaled, labels)

        self.is_fitted = True

    def fit_features(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ):
        """
        Train the detector using precomputed features.
        
        Args:
            X: Feature matrix [n_samples, n_features].
            labels: Binary labels.
        """
        if self.model_type == 'ensemble':
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].fit_transform(X)
                clf.fit(X_scaled, labels)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.classifier.fit(X_scaled, labels)
            
        self.is_fitted = True

    def predict(
        self,
        trackers: List[HypothesisTracker]
    ) -> np.ndarray:
        """
        Predict labels for trackers.

        Args:
            trackers: List of HypothesisTracker instances.

        Returns:
            Predicted labels (0 or 1).
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        X = self.extract_features_from_trackers(trackers)

        if self.model_type == 'ensemble':
            # Average predictions from all classifiers
            predictions = []
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].transform(X)
                pred = clf.predict(X_scaled)
                predictions.append(pred)

            # Majority vote
            predictions = np.array(predictions)
            return (np.mean(predictions, axis=0) > 0.5).astype(int)
        else:
            X_scaled = self.scaler.transform(X)
            return self.classifier.predict(X_scaled)

    def predict_proba(
        self,
        trackers: List[HypothesisTracker]
    ) -> np.ndarray:
        """
        Predict probabilities for trackers.

        Args:
            trackers: List of HypothesisTracker instances.

        Returns:
            Predicted probabilities [n_samples, 2].
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        X = self.extract_features_from_trackers(trackers)

        if self.model_type == 'ensemble':
            # Average probabilities from all classifiers
            proba_list = []
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].transform(X)
                proba = clf.predict_proba(X_scaled)
                proba_list.append(proba)

            return np.mean(proba_list, axis=0)
        else:
            X_scaled = self.scaler.transform(X)
            return self.classifier.predict_proba(X_scaled)

    def evaluate(
        self,
        trackers: List[HypothesisTracker],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate detector on test set.

        Args:
            trackers: List of HypothesisTracker instances.
            labels: True labels.

        Returns:
            Dictionary of metrics.
        """
        # Get predictions
        y_pred = self.predict(trackers)
        y_proba = self.predict_proba(trackers)[:, 1]  # Probability of hallucination

        # Compute metrics
        accuracy = accuracy_score(labels, y_pred)
        auroc = roc_auc_score(labels, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average='binary'
        )

        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        if self.model_type == 'ensemble':
            predictions = []
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].transform(X)
                predictions.append(clf.predict(X_scaled))
            return (np.mean(predictions, axis=0) > 0.5).astype(int)
        else:
            X_scaled = self.scaler.transform(X)
            return self.classifier.predict(X_scaled)

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        if self.model_type == 'ensemble':
            proba_list = []
            for i, clf in enumerate(self.classifiers):
                X_scaled = self.scaler[i].transform(X)
                proba_list.append(clf.predict_proba(X_scaled))
            return np.mean(proba_list, axis=0)
        else:
            X_scaled = self.scaler.transform(X)
            return self.classifier.predict_proba(X_scaled)

    def evaluate_features(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate using precomputed features."""
        y_pred = self.predict_features(X)
        y_proba = self.predict_proba_features(X)[:, 1]
        
        accuracy = accuracy_score(labels, y_pred)
        auroc = roc_auc_score(labels, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, y_pred, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances (if available).

        Returns:
            Feature importances or None if not available.
        """
        if not self.is_fitted:
            return None

        if self.model_type in ['random_forest', 'gradient_boosting']:
            return self.classifier.feature_importances_
        elif self.model_type == 'logistic_regression':
            return np.abs(self.classifier.coef_[0])
        else:
            return None

    def save(self, path: str):
        """
        Save detector to disk.

        Args:
            path: Path to save detector.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted detector")

        save_dict = {
            'model_type': self.model_type,
            'num_layers': self.num_layers,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }

        if self.model_type == 'ensemble':
            save_dict['classifiers'] = self.classifiers
        else:
            save_dict['classifier'] = self.classifier

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str) -> 'HallucinationDetector':
        """
        Load detector from disk.

        Args:
            path: Path to saved detector.

        Returns:
            HallucinationDetector instance.
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        detector = cls(
            model_type=save_dict['model_type'],
            num_layers=save_dict['num_layers']
        )

        detector.scaler = save_dict['scaler']
        detector.is_fitted = save_dict['is_fitted']

        if detector.model_type == 'ensemble':
            detector.classifiers = save_dict['classifiers']
        else:
            detector.classifier = save_dict['classifier']

        return detector

    def get_feature_names(self) -> List[str]:
        """
        Get names of features used by detector.

        Returns:
            List of feature names.
        """
        return self.metrics_computer.get_feature_names()


def train_detector(
    train_trackers: List[HypothesisTracker],
    train_labels: np.ndarray,
    model_type: str = 'random_forest',
    num_layers: int = 12,
    **model_kwargs
) -> HallucinationDetector:
    """
    Convenience function to train a detector.

    Args:
        train_trackers: Training trackers.
        train_labels: Training labels (0=factual, 1=hallucinated).
        model_type: Type of classifier.
        num_layers: Number of layers.
        **model_kwargs: Additional model arguments.

    Returns:
        Trained HallucinationDetector.
    """
    detector = HallucinationDetector(
        model_type=model_type,
        num_layers=num_layers,
        **model_kwargs
    )

    detector.fit(train_trackers, train_labels)

    return detector
