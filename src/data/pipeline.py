"""
Data Pipeline
=============
Market data collection and feature computation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    dates: List[datetime]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    returns: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': self.dates,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'returns': self.returns
        }).set_index('date')


class MarketDataCollector:
    """
    Collects market data from various sources.

    Primary source: Yahoo Finance (SPY for S&P 500)
    Backup: FRED API for economic indicators
    """

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.cache: Dict[str, pd.DataFrame] = {}

    def fetch(self, start_date: str, end_date: str) -> MarketData:
        """
        Fetch market data for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            MarketData object
        """
        cache_key = f"{self.symbol}_{start_date}_{end_date}"

        if cache_key not in self.cache:
            if HAS_YFINANCE:
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(start=start_date, end=end_date)
            else:
                # Generate synthetic data for demo
                df = self._generate_synthetic_data(start_date, end_date)
            self.cache[cache_key] = df

        df = self.cache[cache_key].copy()

        # Calculate returns
        returns = df['Close'].pct_change().fillna(0).values

        return MarketData(
            symbol=self.symbol,
            dates=df.index.tolist(),
            open=df['Open'].values,
            high=df['High'].values,
            low=df['Low'].values,
            close=df['Close'].values,
            volume=df['Volume'].values,
            returns=returns
        )

    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic market data for demo when yfinance unavailable."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days

        np.random.seed(42)
        n = len(dates)

        # Generate realistic price series with occasional crashes
        returns = np.random.normal(0.0005, 0.012, n)  # Daily returns

        # Add some crash events (~5% of days have >3% drops)
        crash_indices = np.random.choice(n, size=int(n * 0.03), replace=False)
        returns[crash_indices] = np.random.uniform(-0.08, -0.03, len(crash_indices))

        # Build price series
        prices = 400 * np.exp(np.cumsum(returns))  # Start around SPY price

        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'High': prices * (1 + np.random.uniform(0, 0.015, n)),
            'Low': prices * (1 - np.random.uniform(0, 0.015, n)),
            'Close': prices,
            'Volume': np.random.randint(50_000_000, 150_000_000, n)
        }, index=dates)

        return df

    def identify_crashes(self, data: MarketData, threshold: float = -0.05) -> pd.DataFrame:
        """
        Identify crash events in market data.

        Args:
            data: MarketData object
            threshold: Return threshold for crash (e.g., -0.05 = -5%)

        Returns:
            DataFrame with crash dates and severities
        """
        crashes = []

        for i, (date, ret) in enumerate(zip(data.dates, data.returns)):
            if ret < threshold:
                crashes.append({
                    'date': date,
                    'return': ret,
                    'severity': abs(ret),
                    'index': i
                })

        return pd.DataFrame(crashes)


class FeatureEngine:
    """
    Computes the 5 core features for crash prediction.

    Features:
    1. western_aspects: Planetary aspect stress (simplified proxy)
    2. eclipse_cycles: Eclipse proximity indicator
    3. ashtakavarga: Vedic strength score (simplified)
    4. bradley_siderograph: Declination-based indicator
    5. fibonacci_time: Golden ratio time cycles
    """

    # Simplified feature calculation (production would use ephemeris)

    def __init__(self):
        self.feature_names = [
            'western_aspects',
            'eclipse_cycles',
            'ashtakavarga',
            'bradley_siderograph',
            'fibonacci_time'
        ]

    def compute_features(self, date: datetime, market_data: Optional[MarketData] = None) -> np.ndarray:
        """
        Compute all features for a given date.

        Args:
            date: Target date
            market_data: Optional market data for context

        Returns:
            Feature vector (5,)
        """
        features = np.array([
            self._compute_western_aspects(date),
            self._compute_eclipse_cycles(date),
            self._compute_ashtakavarga(date),
            self._compute_bradley_siderograph(date),
            self._compute_fibonacci_time(date)
        ])

        return features

    def compute_features_batch(self, dates: List[datetime],
                                market_data: Optional[MarketData] = None) -> np.ndarray:
        """
        Compute features for multiple dates.

        Returns:
            Feature matrix (n_dates, 5)
        """
        return np.array([self.compute_features(d, market_data) for d in dates])

    def _compute_western_aspects(self, date: datetime) -> float:
        """
        Western planetary aspects stress indicator.

        Simplified: Uses day-of-year cycle as proxy
        Production: Would use Swiss Ephemeris for actual aspects
        """
        day_of_year = date.timetuple().tm_yday
        # Multi-cycle combination
        cycle1 = np.sin(2 * np.pi * day_of_year / 365.25)
        cycle2 = np.sin(2 * np.pi * day_of_year / 29.5)  # Lunar month
        cycle3 = np.sin(2 * np.pi * day_of_year / 84)    # Uranus cycle proxy

        return (cycle1 * 0.5 + cycle2 * 0.3 + cycle3 * 0.2 + 1) / 2

    def _compute_eclipse_cycles(self, date: datetime) -> float:
        """
        Eclipse cycle proximity indicator.

        Based on Saros cycle (~18.03 years / 223 synodic months)
        """
        # Reference eclipse date
        ref_eclipse = datetime(2024, 4, 8)  # Recent total solar eclipse
        days_since = (date - ref_eclipse).days

        # Saros cycle: ~6585.3 days
        saros_days = 6585.3
        phase = (days_since % saros_days) / saros_days

        # Higher values near eclipse windows
        # Eclipse windows occur at phase ~0, ~0.5
        eclipse_proximity = 1 - min(phase, 1 - phase) * 2
        return np.clip(eclipse_proximity, 0, 1)

    def _compute_ashtakavarga(self, date: datetime) -> float:
        """
        Vedic Ashtakavarga strength indicator.

        Simplified: Uses lunar cycle as proxy
        Production: Would compute actual bindus
        """
        day_of_year = date.timetuple().tm_yday
        # Lunar mansions (27 nakshatras)
        nakshatra_cycle = np.sin(2 * np.pi * day_of_year / 27.3)
        # Saturn cycle influence (29.5 years simplified)
        saturn_cycle = np.sin(2 * np.pi * (date.year + day_of_year/365) / 29.5)

        return (nakshatra_cycle * 0.6 + saturn_cycle * 0.4 + 1) / 2

    def _compute_bradley_siderograph(self, date: datetime) -> float:
        """
        Bradley Siderograph indicator.

        Based on planetary declinations and aspects.
        Simplified: Seasonal + Jupiter cycle proxy
        """
        day_of_year = date.timetuple().tm_yday
        # Seasonal component
        seasonal = np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Peak ~March equinox
        # Jupiter cycle (~12 years)
        jupiter = np.sin(2 * np.pi * (date.year + day_of_year/365) / 11.86)

        return (seasonal * 0.5 + jupiter * 0.5 + 1) / 2

    def _compute_fibonacci_time(self, date: datetime) -> float:
        """
        Fibonacci time cycle indicator.

        Based on golden ratio time relationships from major market events.
        """
        # Reference points: Major market bottoms
        ref_dates = [
            datetime(2009, 3, 9),   # GFC bottom
            datetime(2020, 3, 23),  # COVID bottom
            datetime(2022, 10, 12), # 2022 bottom
        ]

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        max_days = 2000  # Look back window

        score = 0
        for ref in ref_dates:
            days_since = (date - ref).days
            if 0 < days_since < max_days:
                for ratio in fib_ratios:
                    fib_day = int(days_since * ratio)
                    # Check if current day is near a Fibonacci time projection
                    proximity = abs(days_since - fib_day) / max(fib_day, 1)
                    if proximity < 0.05:  # Within 5% of Fib level
                        score += (1 - proximity) * 0.2

        return np.clip(score, 0, 1)


class CrashDataset:
    """
    Builds training dataset for crash prediction.
    """

    def __init__(self, collector: MarketDataCollector, feature_engine: FeatureEngine):
        self.collector = collector
        self.feature_engine = feature_engine

    def build(self, start_date: str, end_date: str,
              crash_threshold: float = -0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
        """
        Build training dataset.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            crash_threshold: Return threshold for crash classification

        Returns:
            Tuple of (X_features, y_severity, y_timing, dates)
        """
        # Fetch market data
        market_data = self.collector.fetch(start_date, end_date)
        dates = market_data.dates

        # Compute features
        X = self.feature_engine.compute_features_batch(dates, market_data)

        # Targets
        y_severity = market_data.returns  # Continuous
        y_timing = (market_data.returns < crash_threshold).astype(int)  # Binary

        return X, y_severity, y_timing, dates

    def get_statistics(self, X: np.ndarray, y_severity: np.ndarray,
                       y_timing: np.ndarray) -> Dict:
        """Get dataset statistics."""
        return {
            'n_samples': len(y_severity),
            'n_crashes': int(y_timing.sum()),
            'crash_rate': float(y_timing.mean()),
            'avg_return': float(y_severity.mean()),
            'std_return': float(y_severity.std()),
            'min_return': float(y_severity.min()),
            'max_return': float(y_severity.max()),
            'feature_means': X.mean(axis=0).tolist(),
            'feature_stds': X.std(axis=0).tolist()
        }


if __name__ == "__main__":
    print("=== Data Pipeline Demo ===\n")

    # Initialize
    collector = MarketDataCollector("SPY")
    feature_engine = FeatureEngine()
    dataset_builder = CrashDataset(collector, feature_engine)

    # Build dataset (last 2 years)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    print(f"Building dataset from {start_date} to {end_date}...")
    X, y_sev, y_tim, dates = dataset_builder.build(start_date, end_date)

    # Statistics
    stats = dataset_builder.get_statistics(X, y_sev, y_tim)
    print("\nDataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sample features
    print("\nSample features (last 5 days):")
    for i in range(-5, 0):
        print(f"  {dates[i].strftime('%Y-%m-%d')}: {X[i].round(3)}")

    # Identify crashes
    market_data = collector.fetch(start_date, end_date)
    crashes = collector.identify_crashes(market_data)
    print(f"\nCrashes identified: {len(crashes)}")
    if len(crashes) > 0:
        print(crashes.head(10))
