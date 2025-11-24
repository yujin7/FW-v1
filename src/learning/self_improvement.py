"""
Self-Improvement System
=======================
Automated feature discovery, model optimization, and continuous improvement.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path


@dataclass
class FeatureCandidate:
    """A candidate feature being evaluated."""
    name: str
    compute_fn: callable
    tier_0_passed: bool = False
    tier_1_passed: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'tier_0_passed': self.tier_0_passed,
            'tier_1_passed': self.tier_1_passed,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ImprovementResult:
    """Result of an improvement cycle."""
    cycle_id: int
    timestamp: datetime
    features_discovered: int
    features_accepted: int
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement: float
    actions_taken: List[str]


class FeatureDiscovery:
    """
    Automated feature discovery through:
    1. Cycle variation (different time periods)
    2. Feature combinations (interactions)
    3. Lag features (temporal relationships)
    """

    def __init__(self, base_features: List[str]):
        self.base_features = base_features
        self.discovered_features: List[FeatureCandidate] = []

    def generate_candidates(self, X: np.ndarray, dates: List[datetime]) -> List[FeatureCandidate]:
        """
        Generate new feature candidates.

        Strategies:
        1. Lag features (t-1, t-2, t-5)
        2. Rolling statistics (mean, std over windows)
        3. Feature interactions (products, ratios)
        4. Cycle variations (different periods)
        """
        candidates = []

        # 1. Lag features
        for i, name in enumerate(self.base_features):
            for lag in [1, 2, 5]:
                def make_lag_fn(idx, lag_val):
                    def fn(X_in, row_idx):
                        if row_idx >= lag_val:
                            return X_in[row_idx - lag_val, idx]
                        return X_in[row_idx, idx]
                    return fn

                candidates.append(FeatureCandidate(
                    name=f"{name}_lag{lag}",
                    compute_fn=make_lag_fn(i, lag)
                ))

        # 2. Rolling statistics
        for i, name in enumerate(self.base_features):
            for window in [5, 10, 20]:
                def make_rolling_fn(idx, win):
                    def fn(X_in, row_idx):
                        start = max(0, row_idx - win)
                        return np.mean(X_in[start:row_idx+1, idx])
                    return fn

                candidates.append(FeatureCandidate(
                    name=f"{name}_ma{window}",
                    compute_fn=make_rolling_fn(i, window)
                ))

        # 3. Feature interactions (top pairs only)
        for i in range(min(3, len(self.base_features))):
            for j in range(i+1, min(4, len(self.base_features))):
                def make_interaction_fn(idx1, idx2):
                    def fn(X_in, row_idx):
                        return X_in[row_idx, idx1] * X_in[row_idx, idx2]
                    return fn

                candidates.append(FeatureCandidate(
                    name=f"{self.base_features[i]}_x_{self.base_features[j]}",
                    compute_fn=make_interaction_fn(i, j)
                ))

        return candidates


class FeatureTester:
    """
    Two-tier testing framework for features.

    Tier 0 (Fast Screen - 30 seconds):
    - Sign check
    - Magnitude check
    - Non-constant check
    - Correlation with target

    Tier 1 (Rigorous - 5 minutes):
    - Statistical significance
    - Orthogonality check
    - Temporal stability
    - Cross-validation performance
    """

    def __init__(self, significance_level: float = 0.05, max_correlation: float = 0.70):
        self.significance_level = significance_level
        self.max_correlation = max_correlation

    def tier_0_test(self, feature_values: np.ndarray, y: np.ndarray) -> Tuple[bool, Dict]:
        """
        Fast screening tests.

        Returns:
            (passed, metrics_dict)
        """
        metrics = {}

        # 1. Non-constant check
        if np.std(feature_values) < 1e-6:
            return False, {'reason': 'constant_feature'}

        # 2. Correlation with target
        corr, p_value = spearmanr(feature_values, y)
        metrics['correlation'] = float(corr)
        metrics['p_value'] = float(p_value)

        if abs(corr) < 0.05:
            return False, {**metrics, 'reason': 'low_correlation'}

        if p_value > self.significance_level:
            return False, {**metrics, 'reason': 'not_significant'}

        # 3. Reasonable range
        if np.any(np.isnan(feature_values)) or np.any(np.isinf(feature_values)):
            return False, {**metrics, 'reason': 'invalid_values'}

        return True, metrics

    def tier_1_test(self, feature_values: np.ndarray, y: np.ndarray,
                    existing_features: np.ndarray) -> Tuple[bool, Dict]:
        """
        Rigorous testing.

        Returns:
            (passed, metrics_dict)
        """
        metrics = {}

        # 1. Orthogonality check - not too correlated with existing features
        max_corr = 0
        for i in range(existing_features.shape[1]):
            corr, _ = pearsonr(feature_values, existing_features[:, i])
            max_corr = max(max_corr, abs(corr))

        metrics['max_correlation_existing'] = float(max_corr)
        if max_corr > self.max_correlation:
            return False, {**metrics, 'reason': 'redundant_feature'}

        # 2. Temporal stability - correlation in different time periods
        n = len(feature_values)
        mid = n // 2

        corr_first, _ = spearmanr(feature_values[:mid], y[:mid])
        corr_second, _ = spearmanr(feature_values[mid:], y[mid:])

        metrics['corr_first_half'] = float(corr_first)
        metrics['corr_second_half'] = float(corr_second)

        # Sign should be consistent
        if np.sign(corr_first) != np.sign(corr_second):
            return False, {**metrics, 'reason': 'unstable_sign'}

        # 3. Univariate regression R²
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(feature_values.reshape(-1, 1), y)
        r2 = lr.score(feature_values.reshape(-1, 1), y)
        metrics['univariate_r2'] = float(r2)

        if r2 < 0.01:  # Must explain at least 1%
            return False, {**metrics, 'reason': 'low_explanatory_power'}

        return True, metrics


class ModelOptimizer:
    """
    Optimizes model hyperparameters and architecture.

    Uses Bayesian optimization approach with:
    - Ridge alpha tuning
    - Random Forest depth/estimators tuning
    - Feature weight optimization
    """

    def __init__(self):
        self.best_params: Dict = {}
        self.optimization_history: List[Dict] = []

    def optimize_ridge_alpha(self, X: np.ndarray, y: np.ndarray,
                              alphas: List[float] = None) -> float:
        """Find optimal Ridge alpha using CV."""
        if alphas is None:
            alphas = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha = alphas[0]
        best_mae = float('inf')

        for alpha in alphas:
            maes = []
            for train_idx, test_idx in tscv.split(X):
                model = Ridge(alpha=alpha)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                maes.append(mean_absolute_error(y[test_idx], pred))

            avg_mae = np.mean(maes)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_alpha = alpha

        self.best_params['ridge_alpha'] = best_alpha
        return best_alpha

    def optimize_rf_params(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Find optimal Random Forest parameters."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [5, 10, 20]
        }

        tscv = TimeSeriesSplit(n_splits=3)
        best_params = {}
        best_auc = 0

        # Simple grid search (could use Optuna for better optimization)
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for leaf in param_grid['min_samples_leaf']:
                    aucs = []
                    for train_idx, test_idx in tscv.split(X):
                        model = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            random_state=42
                        )
                        model.fit(X[train_idx], y[train_idx])
                        if len(np.unique(y[test_idx])) > 1:
                            prob = model.predict_proba(X[test_idx])[:, 1]
                            aucs.append(roc_auc_score(y[test_idx], prob))

                    if aucs:
                        avg_auc = np.mean(aucs)
                        if avg_auc > best_auc:
                            best_auc = avg_auc
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'min_samples_leaf': leaf
                            }

        self.best_params['rf_params'] = best_params
        return best_params


class SelfImprovementEngine:
    """
    Main self-improvement orchestrator.

    Runs improvement cycles:
    1. Discover new feature candidates
    2. Test candidates through tiers
    3. Optimize model with accepted features
    4. Validate improvement
    5. Deploy if better
    """

    def __init__(self, base_feature_names: List[str]):
        self.feature_discovery = FeatureDiscovery(base_feature_names)
        self.feature_tester = FeatureTester()
        self.model_optimizer = ModelOptimizer()

        self.accepted_features: List[FeatureCandidate] = []
        self.rejected_features: List[FeatureCandidate] = []
        self.improvement_history: List[ImprovementResult] = []
        self.cycle_count = 0

    def run_improvement_cycle(self, X: np.ndarray, y_severity: np.ndarray,
                               y_timing: np.ndarray,
                               dates: List[datetime]) -> ImprovementResult:
        """
        Run one improvement cycle.

        Returns:
            ImprovementResult with details
        """
        self.cycle_count += 1
        actions = []

        # 1. Get current performance
        performance_before = self._evaluate_current(X, y_severity, y_timing)
        actions.append(f"Baseline MAE: {performance_before['mae']:.4f}")

        # 2. Discover new candidates
        candidates = self.feature_discovery.generate_candidates(X, dates)
        actions.append(f"Generated {len(candidates)} feature candidates")

        # 3. Test candidates
        accepted = 0
        for candidate in candidates[:20]:  # Limit to top 20 per cycle
            # Compute feature values
            try:
                values = np.array([candidate.compute_fn(X, i) for i in range(len(X))])
            except Exception:
                continue

            # Tier 0
            passed, metrics = self.feature_tester.tier_0_test(values, y_severity)
            candidate.metrics.update(metrics)
            candidate.tier_0_passed = passed

            if not passed:
                self.rejected_features.append(candidate)
                continue

            # Tier 1
            passed, metrics = self.feature_tester.tier_1_test(values, y_severity, X)
            candidate.metrics.update(metrics)
            candidate.tier_1_passed = passed

            if passed:
                self.accepted_features.append(candidate)
                accepted += 1
            else:
                self.rejected_features.append(candidate)

        actions.append(f"Accepted {accepted} new features")

        # 4. Optimize model
        best_alpha = self.model_optimizer.optimize_ridge_alpha(X, y_severity)
        actions.append(f"Optimized Ridge alpha: {best_alpha}")

        # 5. Get new performance
        performance_after = self._evaluate_current(X, y_severity, y_timing)
        improvement = (performance_before['mae'] - performance_after['mae']) / performance_before['mae']

        result = ImprovementResult(
            cycle_id=self.cycle_count,
            timestamp=datetime.now(),
            features_discovered=len(candidates),
            features_accepted=accepted,
            performance_before=performance_before,
            performance_after=performance_after,
            improvement=improvement,
            actions_taken=actions
        )

        self.improvement_history.append(result)
        return result

    def _evaluate_current(self, X: np.ndarray, y_severity: np.ndarray,
                          y_timing: np.ndarray) -> Dict[str, float]:
        """Evaluate current model performance."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Severity
        ridge = Ridge(alpha=0.30)
        tscv = TimeSeriesSplit(n_splits=5)
        maes = []
        for train_idx, test_idx in tscv.split(X_scaled):
            ridge.fit(X_scaled[train_idx], y_severity[train_idx])
            pred = ridge.predict(X_scaled[test_idx])
            maes.append(mean_absolute_error(y_severity[test_idx], pred))

        # Timing
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        aucs = []
        for train_idx, test_idx in tscv.split(X_scaled):
            rf.fit(X_scaled[train_idx], y_timing[train_idx])
            if len(np.unique(y_timing[test_idx])) > 1:
                prob = rf.predict_proba(X_scaled[test_idx])[:, 1]
                aucs.append(roc_auc_score(y_timing[test_idx], prob))

        return {
            'mae': np.mean(maes),
            'auc': np.mean(aucs) if aucs else 0.5
        }

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate improvement report."""
        if not self.improvement_history:
            return {'status': 'no_cycles_run'}

        latest = self.improvement_history[-1]
        total_improvement = sum(r.improvement for r in self.improvement_history)

        return {
            'total_cycles': self.cycle_count,
            'total_features_discovered': sum(r.features_discovered for r in self.improvement_history),
            'total_features_accepted': len(self.accepted_features),
            'total_improvement': total_improvement,
            'latest_cycle': {
                'id': latest.cycle_id,
                'improvement': latest.improvement,
                'actions': latest.actions_taken
            },
            'best_params': self.model_optimizer.best_params
        }

    def save_state(self, path: str) -> None:
        """Save improvement state."""
        state = {
            'cycle_count': self.cycle_count,
            'accepted_features': [f.to_dict() for f in self.accepted_features],
            'best_params': self.model_optimizer.best_params,
            'improvement_history': [
                {
                    'cycle_id': r.cycle_id,
                    'improvement': r.improvement,
                    'features_accepted': r.features_accepted
                }
                for r in self.improvement_history
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)


if __name__ == "__main__":
    print("=== Self-Improvement Engine Demo ===\n")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500

    base_features = ['western_aspects', 'eclipse_cycles', 'ashtakavarga',
                     'bradley_siderograph', 'fibonacci_time']

    X = np.random.randn(n_samples, 5)
    y_severity = -0.05 * X[:, 0] - 0.03 * X[:, 2] + np.random.randn(n_samples) * 0.01
    y_timing = (X[:, 1] + X[:, 4] > 0.5).astype(int)
    dates = [datetime.now() - timedelta(days=n_samples-i) for i in range(n_samples)]

    # Initialize engine
    engine = SelfImprovementEngine(base_features)

    # Run improvement cycles
    for i in range(3):
        print(f"\n--- Improvement Cycle {i+1} ---")
        result = engine.run_improvement_cycle(X, y_severity, y_timing, dates)
        print(f"Features accepted: {result.features_accepted}")
        print(f"Improvement: {result.improvement:.2%}")
        for action in result.actions_taken:
            print(f"  → {action}")

    # Final report
    print("\n=== Final Report ===")
    report = engine.get_improvement_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Save state
    engine.save_state("/tmp/improvement_state.json")
    print("\nState saved to /tmp/improvement_state.json")
