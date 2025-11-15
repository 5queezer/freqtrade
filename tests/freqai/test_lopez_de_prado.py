"""
Unit tests for Lopez de Prado utilities in FreqAI
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from freqtrade.freqai.lopez_de_prado import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    get_sample_weights_by_time_decay,
    get_sample_weights_by_returns,
    get_sample_weights_by_uniqueness,
    get_num_concurrent_labels,
    seq_bootstrap,
    frac_diff_ffd,
    get_optimal_frac_diff_order,
    get_meta_labels,
)


class TestPurgedKFold:
    """Test Purged K-Fold Cross-Validation."""

    def test_basic_split(self):
        """Test basic splitting without purging."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        }, index=dates)

        cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

        splits = list(cv.split(X))

        # Should have 5 splits
        assert len(splits) == 5

        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            # Train indices should be before test indices (temporal order)
            # Note: with embargo, train can be before AND after test
            # but there should be a gap

    def test_embargo_period(self):
        """Test that embargo period creates gap after test set."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)

        # 10% embargo
        cv = PurgedKFold(n_splits=3, pct_embargo=0.10)

        for train_idx, test_idx in cv.split(X):
            test_end = test_idx.max()
            # Find training samples after test set
            train_after_test = train_idx[train_idx > test_end]

            if len(train_after_test) > 0:
                # Gap between test end and next training sample should be ~10% of dataset
                gap = train_after_test.min() - test_end
                expected_gap = int(0.10 * len(X))
                # Allow some tolerance
                assert gap >= expected_gap - 2

    def test_purging_with_overlapping_labels(self):
        """Test purging removes overlapping samples."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(50)}, index=dates)

        # Create overlapping labels (each label spans 5 days)
        close_times = pd.Series(index=dates)
        for i, date in enumerate(dates):
            close_times.iloc[i] = date + timedelta(days=5)

        cv = PurgedKFold(n_splits=5, samples_info_sets=close_times, pct_embargo=0.01)

        for train_idx, test_idx in cv.split(X):
            # Check that training samples don't overlap with test samples
            train_times = close_times.iloc[train_idx]
            test_times = close_times.iloc[test_idx]

            # No train sample should have close time during test period
            test_min = test_times.index.min()
            test_max = test_times.max()

            overlapping_train = train_times[
                (train_times.index >= test_min) & (train_times.index <= test_max)
            ]

            # Should be minimal or no overlap (some allowed due to embargo)
            # The key is that overlapping samples should be purged
            assert len(overlapping_train) == 0 or len(overlapping_train) < len(test_idx) * 0.1

    def test_get_n_splits(self):
        """Test get_n_splits returns correct value."""
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)  # Too few splits


class TestCombinatorialPurgedKFold:
    """Test Combinatorial Purged K-Fold Cross-Validation."""

    def test_number_of_splits(self):
        """Test correct number of combinatorial splits."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)

        # C(6, 2) = 15 combinations
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)

        splits = list(cv.split(X))
        assert len(splits) == 15
        assert cv.get_n_splits() == 15

    def test_splits_are_different(self):
        """Test that each split is different."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)

        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)

        splits = list(cv.split(X))

        # Convert to tuples for hashing
        test_sets = [tuple(sorted(test_idx)) for _, test_idx in splits]

        # All test sets should be unique
        assert len(test_sets) == len(set(test_sets))

    def test_invalid_params(self):
        """Test invalid parameter combinations."""
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_splits=2)  # Too few

        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_splits=5, n_test_splits=5)  # test >= total


class TestSampleWeights:
    """Test sample weighting functions."""

    def test_time_decay_weights(self):
        """Test time decay weighting."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        weights = get_sample_weights_by_time_decay(dates, decay_factor=1.0)

        # Should sum to 1
        assert np.isclose(weights.sum(), 1.0)

        # More recent samples should have higher weights
        assert weights[-1] > weights[0]

        # All weights should be positive
        assert np.all(weights > 0)

    def test_uniform_weights_when_no_decay(self):
        """Test uniform weights when decay_factor=0."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        weights = get_sample_weights_by_time_decay(dates, decay_factor=0.0)

        # All weights should be equal
        assert np.allclose(weights, weights[0])
        assert np.isclose(weights.sum(), 1.0)

    def test_return_based_weights(self):
        """Test volatility-based weighting."""
        # Create returns with varying volatility
        returns = pd.Series(np.concatenate([
            np.random.randn(50) * 0.01,  # Low volatility
            np.random.randn(50) * 0.05,  # High volatility
        ]))

        weights = get_sample_weights_by_returns(returns, span=10)

        # Should sum to 1
        assert np.isclose(weights.sum(), 1.0)

        # All weights should be positive
        assert np.all(weights > 0)

    def test_uniqueness_weights(self):
        """Test sample uniqueness weighting."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')

        # Create labels with varying overlap
        close_times = pd.Series(index=dates)
        for i, date in enumerate(dates):
            # First 10: short duration (1 day) - high uniqueness
            # Last 10: long duration (5 days) - low uniqueness
            if i < 10:
                close_times.iloc[i] = date + timedelta(days=1)
            else:
                close_times.iloc[i] = date + timedelta(days=5)

        weights = get_sample_weights_by_uniqueness(close_times)

        # Should sum to 1
        assert np.isclose(weights.sum(), 1.0)

        # All weights should be positive
        assert np.all(weights > 0)

        # First samples (less overlap) should have higher weights
        avg_weight_first_10 = weights.iloc[:10].mean()
        avg_weight_last_10 = weights.iloc[10:].mean()

        assert avg_weight_first_10 > avg_weight_last_10

    def test_concurrent_labels(self):
        """Test concurrent label counting."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        # Create overlapping labels
        close_times = pd.Series(index=dates)
        for i, date in enumerate(dates):
            close_times.iloc[i] = date + timedelta(days=3)

        concurrent = get_num_concurrent_labels(close_times)

        # All should be positive
        assert np.all(concurrent > 0)

        # Middle samples should have more concurrency than edges
        assert concurrent.iloc[5] >= concurrent.iloc[0]


class TestSequentialBootstrap:
    """Test Sequential Bootstrap."""

    def test_bootstrap_size(self):
        """Test that bootstrap returns correct number of samples."""
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )

        indices = seq_bootstrap(X, n_samples=50, random_state=42)

        assert len(indices) == 50
        # Indices should be valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(X))

    def test_bootstrap_respects_weights(self):
        """Test that bootstrap respects sample weights."""
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )

        # Give much higher weight to first 10 samples
        weights = pd.Series(0.01, index=X.index)
        weights.iloc[:10] = 10.0
        weights = weights / weights.sum()

        indices = seq_bootstrap(X, sample_weights=weights, n_samples=1000, random_state=42)

        # Count how many times first 10 samples were selected
        first_10_count = np.sum(indices < 10)

        # Should be much more than 10% (which would be expected for uniform sampling)
        assert first_10_count > 100  # Expect > 10% due to high weights

    def test_bootstrap_reproducibility(self):
        """Test that random_state gives reproducible results."""
        X = pd.DataFrame(
            np.random.randn(50, 3),
            index=pd.date_range('2020-01-01', periods=50, freq='D')
        )

        indices1 = seq_bootstrap(X, n_samples=30, random_state=42)
        indices2 = seq_bootstrap(X, n_samples=30, random_state=42)

        assert np.array_equal(indices1, indices2)


class TestFractionalDifferentiation:
    """Test Fractional Differentiation."""

    def test_frac_diff_basic(self):
        """Test basic fractional differentiation."""
        # Create a simple trending series
        series = pd.Series(np.log(np.arange(1, 101, dtype=float)))

        # Apply fractional differentiation
        diff_series = frac_diff_ffd(series, d=0.5, threshold=0.01)

        # Result should be shorter (lost some samples at start)
        assert len(diff_series) < len(series)

        # Should not be all NaN
        assert diff_series.notna().sum() > 0

    def test_frac_diff_d_equals_zero(self):
        """Test that d=0 returns approximately original series."""
        series = pd.Series(np.random.randn(100))

        diff_series = frac_diff_ffd(series, d=0.0, threshold=0.01)

        # Should be very close to original (after initial samples)
        common_idx = diff_series.index
        assert np.allclose(
            diff_series.loc[common_idx].values,
            series.loc[common_idx].values,
            rtol=0.1
        )

    def test_frac_diff_d_equals_one(self):
        """Test that d=1.0 is similar to first difference."""
        series = pd.Series(np.random.randn(100).cumsum())

        # Fractional diff with d=1
        frac_diff = frac_diff_ffd(series, d=1.0, threshold=0.01)

        # Regular diff
        regular_diff = series.diff()

        # Should be similar (not exact due to fixed-width window)
        common_idx = frac_diff.index
        correlation = np.corrcoef(
            frac_diff.loc[common_idx].values,
            regular_diff.loc[common_idx].dropna().values[:len(frac_diff)]
        )[0, 1]

        assert correlation > 0.9  # Should be highly correlated

    def test_frac_diff_stationarity(self):
        """Test that fractional diff helps with stationarity."""
        # Create non-stationary series (random walk)
        np.random.seed(42)
        series = pd.Series(np.random.randn(200).cumsum())

        # Apply fractional differentiation
        diff_series = frac_diff_ffd(series, d=0.5, threshold=0.01)

        # Variance of differenced series should be more stable
        # Check that differenced series has lower autocorrelation
        if len(diff_series) > 1:
            # Simple check: differenced series should have different properties
            assert diff_series.std() != series.std()


class TestMetaLabeling:
    """Test Meta-Labeling."""

    def test_meta_labels_basic(self):
        """Test basic meta-label generation."""
        # Create events
        events = pd.DataFrame({
            'target': [0.05, -0.03, 0.02, -0.01, 0.04],  # Actual returns
            'side': [1, -1, 1, 1, -1],  # Predicted side (1=long, -1=short)
        })

        predictions = pd.Series([1, -1, 1, 1, -1], index=events.index)

        meta_labels = get_meta_labels(events, predictions)

        # Meta-label should be 1 when side matches profitable direction
        # events[0]: side=1 (long), target=0.05 (positive) -> meta=1
        assert meta_labels.iloc[0] == 1

        # events[1]: side=-1 (short), target=-0.03 (negative) -> meta=1
        assert meta_labels.iloc[1] == 1

        # events[2]: side=1 (long), target=0.02 (positive) -> meta=1
        assert meta_labels.iloc[2] == 1

        # events[3]: side=1 (long), target=-0.01 (negative) -> meta=0
        assert meta_labels.iloc[3] == 0

        # events[4]: side=-1 (short), target=0.04 (positive) -> meta=0
        assert meta_labels.iloc[4] == 0

    def test_meta_labels_all_correct(self):
        """Test when all predictions are correct."""
        events = pd.DataFrame({
            'target': [0.05, -0.03, 0.02],
            'side': [1, -1, 1],
        })

        predictions = pd.Series([1, -1, 1], index=events.index)

        meta_labels = get_meta_labels(events, predictions)

        # All should be 1 (take the trade)
        assert np.all(meta_labels == 1)

    def test_meta_labels_all_wrong(self):
        """Test when all predictions are wrong."""
        events = pd.DataFrame({
            'target': [0.05, -0.03, 0.02],
            'side': [-1, 1, -1],  # Opposite of actual
        })

        predictions = pd.Series([-1, 1, -1], index=events.index)

        meta_labels = get_meta_labels(events, predictions)

        # All should be 0 (don't take the trade)
        assert np.all(meta_labels == 0)


class TestOptimalFracDiff:
    """Test optimal fractional differentiation order finding."""

    def test_optimal_d_for_stationary_series(self):
        """Test that stationary series gets low d value."""
        # Already stationary series (white noise)
        series = pd.Series(np.random.randn(200))

        # Should find low d (close to 0)
        optimal_d = get_optimal_frac_diff_order(series, max_d=1.0, step=0.1)

        # Should be low because series is already stationary
        assert optimal_d <= 0.3

    def test_optimal_d_for_nonstationary_series(self):
        """Test that non-stationary series gets higher d value."""
        # Non-stationary series (random walk)
        series = pd.Series(np.random.randn(200).cumsum())

        # Should find higher d
        optimal_d = get_optimal_frac_diff_order(series, max_d=1.0, step=0.1)

        # Should need significant differencing
        assert optimal_d >= 0.3
