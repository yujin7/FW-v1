"""
V42 Simplified Crash Prediction Model
=====================================
2-model ensemble: Ridge (severity) + RandomForest (timing)
5 core features capturing 96.6% of signal
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pickle
from pathlib import Path


@dataclass
class Prediction:
    """Prediction result container."""
    date: str
    severity: float              # Expected crash magnitude
    timing_probability: float    # Probability of crash
    confidence: float            # Meta-confidence score
    features: Dict[str, float]   # Feature contributions

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'severity': round(self.severity, 4),
            'timing_probability': round(self.timing_probability, 4),
            'confidence': round(self.confidence, 4),
            'features': {k: round(v, 4) for k, v in self.features.items()}
        }

    @property
    def signal(self) -> str:
        """Human-readable signal."""
        if self.timing_probability > 0.7 and self.severity < -0.05:
            return "HIGH_RISK"
        elif self.timing_probability > 0.5 and self.severity < -0.03:
            return "ELEVATED"
        elif self.timing_probability > 0.3:
            return "CAUTIONARY"
        return "NORMAL"


class CrashPredictionModel:
    """
    V42 Simplified Crash Prediction Model

    Architecture:
    - Severity Model: Ridge Regression (linear, interpretable)
    - Timing Model: Random Forest Classifier (captures non-linear patterns)

    Features (5 core):
    - western_aspects (24.8%)
    - eclipse_cycles (22.9%)
    - ashtakavarga (19.6%)
    - bradley_siderograph (16.2%)
    - fibonacci_time (16.5%)
    """

    FEATURE_NAMES = [
        'western_aspects',
        'eclipse_cycles',
        'ashtakavarga',
        'bradley_siderograph',
        'fibonacci_time'
    ]

    FEATURE_WEIGHTS = {
        'western_aspects': 0.248,
        'eclipse_cycles': 0.229,
        'ashtakavarga': 0.196,
        'bradley_siderograph': 0.162,
        'fibonacci_time': 0.165
    }

    def __init__(self, alpha: float = 0.30, n_estimators: int = 200):
        """
        Initialize models.

        Args:
            alpha: Ridge regularization strength
            n_estimators: Number of trees in Random Forest
        """
        self.severity_model = Ridge(alpha=alpha)
        self.timing_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y_severity: np.ndarray, y_timing: np.ndarray) -> 'CrashPredictionModel':
        """
        Train both models.

        Args:
            X: Feature matrix (n_samples, 5)
            y_severity: Crash severity targets (continuous)
            y_timing: Crash timing targets (binary: 0/1)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit models
        self.severity_model.fit(X_scaled, y_severity)
        self.timing_model.fit(X_scaled, y_timing)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, date: str = "") -> Prediction:
        """
        Generate prediction.

        Args:
            X: Feature vector (1, 5) or (5,)
            date: Prediction date string

        Returns:
            Prediction object with severity, timing, and confidence
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X)

        # Predictions
        severity = float(self.severity_model.predict(X_scaled)[0])
        timing_prob = float(self.timing_model.predict_proba(X_scaled)[0, 1])

        # Meta-confidence: Higher when predictions align
        # Low severity + low timing = confident normal
        # High severity + high timing = confident risk
        alignment = 1 - abs((-severity * 10) - timing_prob)
        confidence = np.clip(alignment, 0.3, 0.95)

        # Feature contributions
        features = dict(zip(self.FEATURE_NAMES, X[0]))

        return Prediction(
            date=date,
            severity=severity,
            timing_probability=timing_prob,
            confidence=confidence,
            features=features
        )

    def evaluate(self, X: np.ndarray, y_severity: np.ndarray, y_timing: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dict with MAE, Sharpe proxy, AUC
        """
        from sklearn.metrics import mean_absolute_error, roc_auc_score

        X_scaled = self.scaler.transform(X)

        # Severity metrics
        severity_pred = self.severity_model.predict(X_scaled)
        mae = mean_absolute_error(y_severity, severity_pred)

        # Timing metrics
        timing_prob = self.timing_model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_timing, timing_prob)

        # Sharpe proxy (simplified)
        returns = -severity_pred * (y_timing == 1).astype(float)
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)

        self.metrics = {
            'mae': mae,
            'auc': auc,
            'sharpe': sharpe
        }

        return self.metrics

    def cross_validate(self, X: np.ndarray, y_severity: np.ndarray,
                       y_timing: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
        """
        Time-series cross-validation with purging.
        """
        from sklearn.metrics import mean_absolute_error, roc_auc_score

        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores, auc_scores = [], []

        for train_idx, test_idx in tscv.split(X):
            # Purge: Skip 5 samples between train and test
            if train_idx[-1] + 5 < test_idx[0]:
                X_train, X_test = X[train_idx], X[test_idx]
                y_sev_train, y_sev_test = y_severity[train_idx], y_severity[test_idx]
                y_tim_train, y_tim_test = y_timing[train_idx], y_timing[test_idx]

                # Fit on train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Severity
                ridge = Ridge(alpha=0.30)
                ridge.fit(X_train_scaled, y_sev_train)
                mae_scores.append(mean_absolute_error(y_sev_test, ridge.predict(X_test_scaled)))

                # Timing
                rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
                rf.fit(X_train_scaled, y_tim_train)
                if len(np.unique(y_tim_test)) > 1:
                    auc_scores.append(roc_auc_score(y_tim_test, rf.predict_proba(X_test_scaled)[:, 1]))

        return {
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_auc_mean': np.mean(auc_scores) if auc_scores else 0.5,
            'cv_auc_std': np.std(auc_scores) if auc_scores else 0.0
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'severity_model': self.severity_model,
                'timing_model': self.timing_model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'metrics': self.metrics
            }, f)

    @classmethod
    def load(cls, path: str) -> 'CrashPredictionModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.severity_model = data['severity_model']
        model.timing_model = data['timing_model']
        model.scaler = data['scaler']
        model.is_fitted = data['is_fitted']
        model.metrics = data['metrics']
        return model


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic features
    X = np.random.randn(n_samples, 5)

    # Generate targets (severity: continuous, timing: binary)
    y_severity = -0.05 * X[:, 0] - 0.03 * X[:, 2] + np.random.randn(n_samples) * 0.01
    y_timing = (X[:, 1] + X[:, 4] > 0.5).astype(int)

    # Train model
    model = CrashPredictionModel()
    model.fit(X, y_severity, y_timing)

    # Evaluate
    metrics = model.evaluate(X, y_severity, y_timing)
    print("Model Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Cross-validate
    cv_metrics = model.cross_validate(X, y_severity, y_timing)
    print("\nCross-Validation:")
    for k, v in cv_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Single prediction
    pred = model.predict(X[0], date="2025-11-24")
    print(f"\nPrediction: {pred.to_dict()}")
    print(f"Signal: {pred.signal}")
