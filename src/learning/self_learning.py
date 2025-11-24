"""
Self-Learning System
====================
Automated drift detection, adaptive weight updates, and retraining triggers.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
from pathlib import Path


class TriggerPriority(Enum):
    """Retraining trigger priority levels."""
    CRITICAL = "critical"   # Immediate retraining
    HIGH = "high"           # Within 7 days
    MEDIUM = "medium"       # Within 30 days
    LOW = "low"             # Quarterly


@dataclass
class DriftResult:
    """Result of drift detection."""
    detected: bool
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    drift_magnitude: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def severity(self) -> str:
        if self.drift_magnitude > 0.5:
            return "SEVERE"
        elif self.drift_magnitude > 0.3:
            return "MODERATE"
        elif self.drift_magnitude > 0.2:
            return "MILD"
        return "NONE"


@dataclass
class LearningEvent:
    """Record of a learning/adaptation event."""
    event_type: str           # drift_detected, weights_updated, retrained
    trigger_reason: str
    old_value: Any
    new_value: Any
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class DriftDetector:
    """
    Detects performance and feature drift.

    Methods:
    - MAE Drift: Rolling window comparison
    - KS Test: Distribution shift detection
    - ADWIN: Adaptive windowing for concept drift
    """

    def __init__(self, window_size: int = 30, mae_threshold: float = 0.20):
        """
        Args:
            window_size: Number of observations for rolling metrics
            mae_threshold: Relative MAE increase threshold (e.g., 0.20 = 20%)
        """
        self.window_size = window_size
        self.mae_threshold = mae_threshold
        self.baseline_mae: Optional[float] = None
        self.recent_errors: deque = deque(maxlen=window_size)
        self.drift_history: List[DriftResult] = []

    def set_baseline(self, mae: float) -> None:
        """Set baseline MAE from training."""
        self.baseline_mae = mae

    def add_observation(self, predicted: float, actual: float) -> Optional[DriftResult]:
        """
        Add new observation and check for drift.

        Returns:
            DriftResult if drift detected, None otherwise
        """
        error = abs(predicted - actual)
        self.recent_errors.append(error)

        if len(self.recent_errors) < self.window_size // 2:
            return None

        return self.check_mae_drift()

    def check_mae_drift(self) -> Optional[DriftResult]:
        """Check if MAE has drifted from baseline."""
        if self.baseline_mae is None or len(self.recent_errors) == 0:
            return None

        current_mae = np.mean(self.recent_errors)
        relative_change = (current_mae - self.baseline_mae) / (self.baseline_mae + 1e-6)

        drift_detected = relative_change > self.mae_threshold

        result = DriftResult(
            detected=drift_detected,
            metric_name="mae",
            current_value=current_mae,
            baseline_value=self.baseline_mae,
            threshold=self.mae_threshold,
            drift_magnitude=relative_change
        )

        if drift_detected:
            self.drift_history.append(result)

        return result

    def check_feature_drift(self, current_features: np.ndarray,
                            baseline_features: np.ndarray) -> Dict[str, DriftResult]:
        """
        Check for feature distribution drift using KS test.

        Args:
            current_features: Recent feature values (n_samples, n_features)
            baseline_features: Training feature values

        Returns:
            Dict of feature name -> DriftResult
        """
        from scipy.stats import ks_2samp

        results = {}
        feature_names = ['western_aspects', 'eclipse_cycles', 'ashtakavarga',
                         'bradley_siderograph', 'fibonacci_time']

        for i, name in enumerate(feature_names):
            stat, p_value = ks_2samp(current_features[:, i], baseline_features[:, i])

            drift_detected = p_value < 0.01  # 99% confidence

            results[name] = DriftResult(
                detected=drift_detected,
                metric_name=f"feature_{name}",
                current_value=p_value,
                baseline_value=0.01,
                threshold=0.01,
                drift_magnitude=1 - p_value if drift_detected else 0
            )

        return results


class AdaptiveWeightOptimizer:
    """
    Lightweight weight adjustment without full retraining.

    Updates feature weights based on recent prediction performance
    using exponential moving average of feature contributions.
    """

    def __init__(self, learning_rate: float = 0.05, max_change: float = 0.20):
        """
        Args:
            learning_rate: Step size for weight updates
            max_change: Maximum relative weight change per update
        """
        self.learning_rate = learning_rate
        self.max_change = max_change

        # Initial weights from v42 spec
        self.weights = {
            'western_aspects': 0.248,
            'eclipse_cycles': 0.229,
            'ashtakavarga': 0.196,
            'bradley_siderograph': 0.162,
            'fibonacci_time': 0.165
        }

        self.weight_history: List[Dict] = []

    def update_weights(self, feature_errors: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on feature-specific errors.

        Args:
            feature_errors: Dict of feature name -> error contribution

        Returns:
            Updated weights dict
        """
        old_weights = self.weights.copy()

        # Calculate gradient (features with higher error get lower weight)
        total_error = sum(feature_errors.values()) + 1e-6
        gradients = {k: v / total_error for k, v in feature_errors.items()}

        # Update weights
        for feature, gradient in gradients.items():
            if feature in self.weights:
                # Reduce weight for high-error features
                adjustment = -self.learning_rate * (gradient - 1/len(self.weights))

                # Clip to max change
                max_delta = self.weights[feature] * self.max_change
                adjustment = np.clip(adjustment, -max_delta, max_delta)

                self.weights[feature] += adjustment

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        self.weight_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_weights': old_weights,
            'new_weights': self.weights.copy(),
            'feature_errors': feature_errors
        })

        return self.weights


class RetrainingTriggerManager:
    """
    Manages retraining decisions based on multiple triggers.

    Triggers (priority order):
    1. CRITICAL: New crash event (immediate)
    2. HIGH: Performance drift > 20% (within 7 days)
    3. MEDIUM: 90 days since last training (scheduled)
    """

    def __init__(self):
        self.last_training_date: Optional[datetime] = None
        self.last_crash_date: Optional[datetime] = None
        self.pending_triggers: List[Dict] = []

    def check_triggers(self, drift_result: Optional[DriftResult] = None,
                       new_crash: bool = False) -> Optional[Dict]:
        """
        Check all triggers and return highest priority action.

        Returns:
            Dict with trigger info if action needed, None otherwise
        """
        triggers = []

        # CRITICAL: New crash
        if new_crash:
            triggers.append({
                'priority': TriggerPriority.CRITICAL,
                'reason': 'new_crash_event',
                'action': 'retrain_immediately',
                'deadline': datetime.now()
            })

        # HIGH: Performance drift
        if drift_result and drift_result.detected:
            triggers.append({
                'priority': TriggerPriority.HIGH,
                'reason': f'performance_drift_{drift_result.metric_name}',
                'action': 'retrain_within_week',
                'deadline': datetime.now() + timedelta(days=7),
                'drift_magnitude': drift_result.drift_magnitude
            })

        # MEDIUM: Scheduled (90 days)
        if self.last_training_date:
            days_since = (datetime.now() - self.last_training_date).days
            if days_since >= 90:
                triggers.append({
                    'priority': TriggerPriority.MEDIUM,
                    'reason': 'scheduled_quarterly',
                    'action': 'retrain_scheduled',
                    'deadline': datetime.now() + timedelta(days=30),
                    'days_since_training': days_since
                })

        if not triggers:
            return None

        # Return highest priority
        priority_order = [TriggerPriority.CRITICAL, TriggerPriority.HIGH,
                         TriggerPriority.MEDIUM, TriggerPriority.LOW]
        triggers.sort(key=lambda x: priority_order.index(x['priority']))

        return triggers[0]

    def mark_training_complete(self) -> None:
        """Record that training was completed."""
        self.last_training_date = datetime.now()
        self.pending_triggers = []


class SelfLearningSystem:
    """
    Main orchestrator for self-learning capabilities.

    Integrates:
    - Drift detection
    - Adaptive weight optimization
    - Retraining trigger management
    - Event logging
    """

    def __init__(self, model_path: str = "models/current.pkl"):
        self.drift_detector = DriftDetector()
        self.weight_optimizer = AdaptiveWeightOptimizer()
        self.trigger_manager = RetrainingTriggerManager()
        self.model_path = model_path
        self.events: List[LearningEvent] = []
        self.is_active = True

    def initialize(self, baseline_mae: float, training_date: datetime) -> None:
        """Initialize system with baseline metrics."""
        self.drift_detector.set_baseline(baseline_mae)
        self.trigger_manager.last_training_date = training_date

    def process_observation(self, predicted: float, actual: float,
                            features: Dict[str, float],
                            is_crash: bool = False) -> Dict[str, Any]:
        """
        Process new observation and determine actions.

        Args:
            predicted: Model prediction
            actual: Actual observed value
            features: Feature values used for prediction
            is_crash: Whether this was a crash event

        Returns:
            Dict with drift status, triggers, and recommended actions
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'trigger': None,
            'action': None,
            'weights_updated': False
        }

        # 1. Check drift
        drift_result = self.drift_detector.add_observation(predicted, actual)
        if drift_result and drift_result.detected:
            result['drift_detected'] = True
            result['drift_info'] = {
                'severity': drift_result.severity,
                'magnitude': drift_result.drift_magnitude
            }

        # 2. Check triggers
        trigger = self.trigger_manager.check_triggers(drift_result, is_crash)
        if trigger:
            result['trigger'] = trigger
            result['action'] = trigger['action']

            self.events.append(LearningEvent(
                event_type='trigger_activated',
                trigger_reason=trigger['reason'],
                old_value=None,
                new_value=trigger,
                performance_before={'mae': self.drift_detector.baseline_mae}
            ))

        # 3. Lightweight weight update (if no retraining triggered)
        if not trigger and drift_result and drift_result.drift_magnitude > 0.1:
            # Calculate feature-specific errors (simplified)
            feature_errors = {k: abs(predicted - actual) * v
                             for k, v in features.items()}
            new_weights = self.weight_optimizer.update_weights(feature_errors)
            result['weights_updated'] = True
            result['new_weights'] = new_weights

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_active': self.is_active,
            'baseline_mae': self.drift_detector.baseline_mae,
            'current_mae': np.mean(self.drift_detector.recent_errors)
                           if self.drift_detector.recent_errors else None,
            'drift_history_count': len(self.drift_detector.drift_history),
            'current_weights': self.weight_optimizer.weights,
            'last_training': self.trigger_manager.last_training_date.isoformat()
                            if self.trigger_manager.last_training_date else None,
            'events_count': len(self.events)
        }

    def save_state(self, path: str) -> None:
        """Save system state to disk."""
        state = {
            'baseline_mae': self.drift_detector.baseline_mae,
            'weights': self.weight_optimizer.weights,
            'weight_history': self.weight_optimizer.weight_history[-100:],
            'last_training': self.trigger_manager.last_training_date.isoformat()
                            if self.trigger_manager.last_training_date else None,
            'events': [
                {
                    'type': e.event_type,
                    'reason': e.trigger_reason,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in self.events[-100:]
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> 'SelfLearningSystem':
        """Load system state from disk."""
        system = cls()

        with open(path, 'r') as f:
            state = json.load(f)

        if state.get('baseline_mae'):
            system.drift_detector.set_baseline(state['baseline_mae'])

        if state.get('weights'):
            system.weight_optimizer.weights = state['weights']

        if state.get('last_training'):
            system.trigger_manager.last_training_date = datetime.fromisoformat(
                state['last_training']
            )

        return system


if __name__ == "__main__":
    # Demo
    print("=== Self-Learning System Demo ===\n")

    # Initialize
    system = SelfLearningSystem()
    system.initialize(baseline_mae=0.0074, training_date=datetime.now() - timedelta(days=30))

    # Simulate observations
    np.random.seed(42)
    for i in range(50):
        # Simulate prediction error (increasing over time to trigger drift)
        error_scale = 1 + (i / 100)  # Gradually increasing error
        predicted = np.random.randn() * 0.01
        actual = predicted + np.random.randn() * 0.01 * error_scale

        features = {
            'western_aspects': np.random.rand(),
            'eclipse_cycles': np.random.rand(),
            'ashtakavarga': np.random.rand(),
            'bradley_siderograph': np.random.rand(),
            'fibonacci_time': np.random.rand()
        }

        result = system.process_observation(predicted, actual, features)

        if result['drift_detected'] or result['trigger']:
            print(f"Observation {i+1}: {result}")

    # Final status
    print("\n=== Final Status ===")
    status = system.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Save state
    system.save_state("/tmp/learning_state.json")
    print("\nState saved to /tmp/learning_state.json")
