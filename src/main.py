"""
V42 Crash Prediction System - Main Orchestrator
================================================
Brings together all components for a complete self-learning prediction system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

from models.prediction import CrashPredictionModel, Prediction
from learning.self_learning import SelfLearningSystem
from learning.self_improvement import SelfImprovementEngine
from data.pipeline import MarketDataCollector, FeatureEngine, CrashDataset


class CrashPredictionOrchestrator:
    """
    Main orchestrator for the V42 Crash Prediction System.

    Responsibilities:
    - Model training and inference
    - Self-learning management
    - Daily prediction pipeline
    - Performance monitoring
    """

    def __init__(self, model_path: str = "models/v42_model.pkl",
                 state_path: str = "state/learning_state.json"):
        self.model_path = model_path
        self.state_path = state_path

        # Components
        self.model: Optional[CrashPredictionModel] = None
        self.learning_system: Optional[SelfLearningSystem] = None
        self.improvement_engine: Optional[SelfImprovementEngine] = None
        self.collector = MarketDataCollector("SPY")
        self.feature_engine = FeatureEngine()
        self.dataset_builder = CrashDataset(self.collector, self.feature_engine)

        # State
        self.is_initialized = False
        self.last_prediction: Optional[Prediction] = None
        self.training_data: Optional[tuple] = None  # (X, y_sev, y_tim, dates)

    def initialize(self, train_years: int = 5) -> Dict[str, Any]:
        """
        Initialize system: load or train model.

        Args:
            train_years: Years of historical data for training

        Returns:
            Initialization status
        """
        result = {
            'status': 'initializing',
            'model_loaded': False,
            'model_trained': False,
            'learning_initialized': False
        }

        # Try loading existing model
        try:
            self.model = CrashPredictionModel.load(self.model_path)
            result['model_loaded'] = True
            print(f"✓ Model loaded from {self.model_path}")
        except FileNotFoundError:
            print("No existing model found. Training new model...")
            self.model = self._train_model(train_years)
            result['model_trained'] = True
            print("✓ New model trained")

        # Initialize self-learning system
        self.learning_system = SelfLearningSystem(self.model_path)

        if self.model.metrics:
            self.learning_system.initialize(
                baseline_mae=self.model.metrics.get('mae', 0.01),
                training_date=datetime.now()
            )
            result['learning_initialized'] = True
            print("✓ Self-learning system initialized")

        # Initialize self-improvement engine
        self.improvement_engine = SelfImprovementEngine(self.feature_engine.feature_names)
        result['improvement_initialized'] = True
        print("✓ Self-improvement engine initialized")

        self.is_initialized = True
        result['status'] = 'ready'

        return result

    def _train_model(self, train_years: int = 5) -> CrashPredictionModel:
        """Train a new model on historical data."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=train_years * 365)).strftime("%Y-%m-%d")

        print(f"Building dataset from {start_date} to {end_date}...")
        X, y_sev, y_tim, dates = self.dataset_builder.build(start_date, end_date)

        stats = self.dataset_builder.get_statistics(X, y_sev, y_tim)
        print(f"Dataset: {stats['n_samples']} samples, {stats['n_crashes']} crashes")

        # Store for improvement cycles
        self.training_data = (X, y_sev, y_tim, dates)

        # Train
        model = CrashPredictionModel()
        model.fit(X, y_sev, y_tim)

        # Evaluate
        metrics = model.evaluate(X, y_sev, y_tim)
        print(f"Training metrics: MAE={metrics['mae']:.4f}, AUC={metrics['auc']:.4f}")

        # Cross-validate
        cv_metrics = model.cross_validate(X, y_sev, y_tim)
        print(f"CV metrics: MAE={cv_metrics['cv_mae_mean']:.4f}±{cv_metrics['cv_mae_std']:.4f}")

        # Save
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(self.model_path)

        return model

    def predict(self, date: Optional[datetime] = None) -> Prediction:
        """
        Generate prediction for a date.

        Args:
            date: Target date (default: tomorrow)

        Returns:
            Prediction object
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        if date is None:
            date = datetime.now() + timedelta(days=1)

        # Compute features
        features = self.feature_engine.compute_features(date)
        date_str = date.strftime("%Y-%m-%d")

        # Generate prediction
        prediction = self.model.predict(features, date_str)
        self.last_prediction = prediction

        return prediction

    def update_with_observation(self, date: datetime, actual_return: float) -> Dict[str, Any]:
        """
        Process actual observation and update learning system.

        Args:
            date: Observation date
            actual_return: Actual market return

        Returns:
            Learning system response
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized.")

        # Get what we predicted for this date
        features = self.feature_engine.compute_features(date)
        prediction = self.model.predict(features, date.strftime("%Y-%m-%d"))

        is_crash = actual_return < -0.05  # 5% threshold

        # Process through learning system
        learning_result = self.learning_system.process_observation(
            predicted=prediction.severity,
            actual=actual_return,
            features=prediction.features,
            is_crash=is_crash
        )

        # Handle retraining trigger
        if learning_result.get('action') == 'retrain_immediately':
            print("⚠️ Critical trigger: Initiating retraining...")
            self.model = self._train_model(train_years=5)
            self.learning_system.trigger_manager.mark_training_complete()
            learning_result['retrained'] = True

        return learning_result

    def run_daily_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete daily pipeline.

        Steps:
        1. Fetch latest market data
        2. Check for new observations
        3. Generate tomorrow's prediction
        4. Update learning system

        Returns:
            Pipeline results
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'prediction': None,
            'learning_update': None,
            'status': 'running'
        }

        try:
            # 1. Generate prediction for tomorrow
            prediction = self.predict()
            result['prediction'] = prediction.to_dict()
            print(f"\n📈 Prediction for {prediction.date}:")
            print(f"   Severity: {prediction.severity:.4f}")
            print(f"   Timing Probability: {prediction.timing_probability:.4f}")
            print(f"   Signal: {prediction.signal}")

            # 2. Check yesterday's prediction (if available)
            yesterday = datetime.now() - timedelta(days=1)
            market_data = self.collector.fetch(
                yesterday.strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d")
            )

            if len(market_data.returns) > 0:
                actual_return = market_data.returns[-1]
                learning_result = self.update_with_observation(yesterday, actual_return)
                result['learning_update'] = learning_result

                if learning_result.get('drift_detected'):
                    print(f"⚠️ Drift detected: {learning_result.get('drift_info')}")

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"❌ Pipeline error: {e}")

        return result

    def run_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run a self-improvement cycle.

        Discovers new features, tests them, optimizes model.
        """
        if not self.is_initialized or self.training_data is None:
            raise RuntimeError("System not initialized with training data.")

        print("\n🔄 Running self-improvement cycle...")

        X, y_sev, y_tim, dates = self.training_data
        result = self.improvement_engine.run_improvement_cycle(X, y_sev, y_tim, dates)

        print(f"   Features discovered: {result.features_discovered}")
        print(f"   Features accepted: {result.features_accepted}")
        print(f"   Improvement: {result.improvement:.2%}")

        # If significant improvement, retrain model with new insights
        if result.improvement > 0.05:  # >5% improvement
            print("   ✓ Significant improvement! Updating model...")
            self.model = self._train_model(train_years=2)

        return {
            'cycle_id': result.cycle_id,
            'features_discovered': result.features_discovered,
            'features_accepted': result.features_accepted,
            'improvement': result.improvement,
            'actions': result.actions_taken
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'initialized': self.is_initialized,
            'model': None,
            'learning': None,
            'improvement': None,
            'last_prediction': None
        }

        if self.model and self.model.metrics:
            status['model'] = {
                'metrics': self.model.metrics,
                'is_fitted': self.model.is_fitted
            }

        if self.learning_system:
            status['learning'] = self.learning_system.get_status()

        if self.improvement_engine:
            status['improvement'] = self.improvement_engine.get_improvement_report()

        if self.last_prediction:
            status['last_prediction'] = self.last_prediction.to_dict()

        return status

    def save_state(self) -> None:
        """Save all state to disk."""
        if self.model:
            self.model.save(self.model_path)

        if self.learning_system:
            self.learning_system.save_state(self.state_path)

        print(f"✓ State saved")


def main():
    """Main entry point."""
    print("=" * 60)
    print("V42 CRASH PREDICTION SYSTEM")
    print("Self-Learning & Self-Improving Market Crash Predictor")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = CrashPredictionOrchestrator(
        model_path="models/v42_model.pkl",
        state_path="state/learning_state.json"
    )

    # Initialize system
    print("\n[1/4] Initializing system...")
    init_result = orchestrator.initialize(train_years=2)  # Use 2 years for demo
    print(f"Initialization: {init_result['status']}")

    # Run daily pipeline
    print("\n[2/4] Running daily pipeline...")
    pipeline_result = orchestrator.run_daily_pipeline()

    # Run self-improvement cycle
    print("\n[3/4] Running self-improvement cycle...")
    improvement_result = orchestrator.run_improvement_cycle()

    # Show status
    print("\n[4/4] System Status:")
    status = orchestrator.get_status()

    if status['model']:
        print(f"   Model MAE: {status['model']['metrics'].get('mae', 'N/A'):.4f}")
        print(f"   Model AUC: {status['model']['metrics'].get('auc', 'N/A'):.4f}")

    if status['learning']:
        print(f"   Learning baseline MAE: {status['learning'].get('baseline_mae', 'N/A')}")
        print(f"   Events logged: {status['learning'].get('events_count', 0)}")

    if status['improvement']:
        print(f"   Improvement cycles: {status['improvement'].get('total_cycles', 0)}")
        print(f"   Features discovered: {status['improvement'].get('total_features_discovered', 0)}")
        print(f"   Features accepted: {status['improvement'].get('total_features_accepted', 0)}")

    # Save state
    print("\n[Complete] Saving state...")
    orchestrator.save_state()

    print("\n" + "=" * 60)
    print("System ready for production use.")
    print("Features: Self-Learning + Self-Improving")
    print("=" * 60)

    return orchestrator


if __name__ == "__main__":
    main()
