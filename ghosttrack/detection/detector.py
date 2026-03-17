"""
Hallucination detector using track divergence metrics.

:class:`HallucinationDetector` accepts pre-computed feature matrices
(``np.ndarray``) produced by
:meth:`~ghosttrack.metrics.MetricsRegistry.feature_vector`.  This
design separates feature computation from model fitting, making the
class easy to unit-test without a running transformer model.

Supported classifiers
---------------------
``random_forest``, ``gradient_boosting``, ``logistic_regression``,
``svm``, ``ensemble`` (soft-vote average of the first three).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class DetectionMetrics(NamedTuple):
    """Evaluation result for a fitted detector."""

    auroc: float
    accuracy: float
    precision: float
    recall: float
    f1: float


class HallucinationDetector:
    """
    Binary classifier for hallucination detection.

    Args:
        model_type: Classifier family — ``"random_forest"``,
            ``"gradient_boosting"``, ``"logistic_regression"``,
            ``"svm"``, or ``"ensemble"``.
        n_layers: Number of transformer layers (kept for metadata only).
        **kwargs: Extra kwargs forwarded to the sklearn classifier.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_layers: int = 12,
        **kwargs,
    ):
        self.model_type = model_type
        self.n_layers = n_layers
        self._is_fitted = False

        # class_weight='balanced' is critical for TruthfulQA where GPT-2 Medium
        # hallucinates ~75% of the time, leaving far fewer factual examples.
        if model_type == "random_forest":
            self._clf = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth"),
                class_weight="balanced",
                random_state=kwargs.get("random_state", 42),
            )
            self._scaler = StandardScaler()
        elif model_type == "gradient_boosting":
            # GradientBoostingClassifier does not support class_weight;
            # imbalance is handled via sample_weight in fit() if needed.
            self._clf = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 3),
                random_state=kwargs.get("random_state", 42),
            )
            self._scaler = StandardScaler()
        elif model_type == "logistic_regression":
            self._clf = LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 1000),
                class_weight="balanced",
                random_state=kwargs.get("random_state", 42),
            )
            self._scaler = StandardScaler()
        elif model_type == "svm":
            self._clf = SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
                class_weight="balanced",
                random_state=kwargs.get("random_state", 42),
            )
            self._scaler = StandardScaler()
        elif model_type == "ensemble":
            self._ensemble_clfs = [
                RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ]
            self._ensemble_scalers = [StandardScaler() for _ in self._ensemble_clfs]
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    # ------------------------------------------------------------------ #
    # Primary interface (feature-matrix based)                            #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationDetector":
        """
        Fit on pre-computed feature matrix *X*.

        Args:
            X: ``[n_samples, n_features]`` float32 array.
            y: Binary labels ``[n_samples]`` (0 = factual, 1 = hallucinated).

        Returns:
            ``self`` for chaining.
        """
        if self.model_type == "ensemble":
            for clf, sc in zip(self._ensemble_clfs, self._ensemble_scalers):
                clf.fit(sc.fit_transform(X), y)
        else:
            self._clf.fit(self._scaler.fit_transform(X), y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for *X*."""
        self._check_fitted()
        if self.model_type == "ensemble":
            preds = np.stack(
                [clf.predict(sc.transform(X))
                 for clf, sc in zip(self._ensemble_clfs, self._ensemble_scalers)]
            )
            return (preds.mean(axis=0) > 0.5).astype(int)
        return self._clf.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for *X* — shape ``[n_samples, 2]``."""
        self._check_fitted()
        if self.model_type == "ensemble":
            probas = np.stack(
                [clf.predict_proba(sc.transform(X))
                 for clf, sc in zip(self._ensemble_clfs, self._ensemble_scalers)]
            )
            return probas.mean(axis=0)
        return self._clf.predict_proba(self._scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> DetectionMetrics:
        """
        Evaluate the fitted detector.

        Args:
            X: Feature matrix ``[n_samples, n_features]``.
            y: True binary labels.

        Returns:
            :class:`DetectionMetrics` named tuple.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        return DetectionMetrics(
            auroc=float(roc_auc_score(y, y_proba)),
            accuracy=float(accuracy_score(y, y_pred)),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
        )

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances (or ``None`` for SVM / ensemble)."""
        if not self._is_fitted or self.model_type == "ensemble":
            return None
        if self.model_type in ("random_forest", "gradient_boosting"):
            return self._clf.feature_importances_
        if self.model_type == "logistic_regression":
            return np.abs(self._clf.coef_[0])
        return None

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serialise the fitted detector to *path* using pickle."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "HallucinationDetector":
        """Deserialise a detector saved with :meth:`save`."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Detector is not fitted.  Call fit() first.")
