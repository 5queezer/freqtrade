"""
Unit tests for Lopez de Prado utilities in FreqAI
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from freqtrade.freqai.lopez_de_prado import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    frac_diff_ffd,
    get_bins_from_triple_barrier,
    get_events_triple_barrier,
    get_meta_labels,
    get_num_concurrent_labels,
    get_optimal_frac_diff_order,
    get_sample_weights_by_returns,
    get_sample_weights_by_time_decay,
    get_sample_weights_by_uniqueness,
    seq_bootstrap,
)


class TestPurgedKFold:
    """Test Purged K-Fold Cross-Validation."""

    def test_basic_split(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)},
                         index=dates)
        cv = PurgedKFold(n_splits=5, pct_embargo=0.01)
        splits = list(cv.split(X))

        assert len(splits) == 5
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_embargo_period(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)
        cv = PurgedKFold(n_splits=3, pct_embargo=0.10)

        for train_idx, test_idx in cv.split(X):
            test_end = test_idx.max()
            train_after_test = train_idx[train_idx > test_end]
            if len(train_after_test) > 0:
                gap = train_after_test.min() - test_end
                expected_gap = int(0.10 * len(X))
                assert gap >= expected_gap - 2

    def test_purging_with_overlapping_labels(self):
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(50)}, index=dates)
        close_times = pd.Series([date + timedelta(days=5) for date in dates], index=dates)

        cv = PurgedKFold(n_splits=5, samples_info_sets=close_times, pct_embargo=0.01)

        for train_idx, test_idx in cv.split(X):
            train_times = close_times.iloc[train_idx]
            test_times = close_times.iloc[test_idx]
            test_min = test_times.index.min()
            test_max = test_times.max()
            overlapping_train = train_times[
                (train_times.index >= test_min) & (train_times.index <= test_max)
            ]
            assert len(overlapping_train) == 0 or len(overlapping_train) < len(test_idx) * 0.1

    def test_get_n_splits(self):
        cv = PurgedKFold(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)


class TestCombinatorialPurgedKFold:
    """Test Combinatorial Purged K-Fold Cross-Validation."""

    def test_number_of_splits(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)
        cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)

        splits = list(cv.split(X))
        assert len(splits) == 15
        assert cv.get_n_splits() == 15

    def test_splits_are_different(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        X = pd.DataFrame({'feat': np.random.randn(100)}, index=dates)
        cv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2)

        splits = list(cv.split(X))
        test_sets = [tuple(sorted(test_idx)) for _, test_idx in splits]
        assert len(test_sets) == len(set(test_sets))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_splits=2)
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(n_splits=5, n_test_splits=5)


class TestSampleWeights:
    """Test sample weighting functions."""

    def test_time_decay_weights(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        weights = get_sample_weights_by_time_decay(dates, decay_factor=1.0)

        assert np.isclose(weights.sum(), 1.0)
        assert weights[-1] > weights[0]
        assert np.all(weights > 0)

    def test_uniform_weights_when_no_decay(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        weights = get_sample_weights_by_time_decay(dates, decay_factor=0.0)

        assert np.allclose(weights, weights[0])
        assert np.isclose(weights.sum(), 1.0)

    def test_return_based_weights(self):
        returns = pd.Series(np.concatenate([
            np.random.randn(50) * 0.01,
            np.random.randn(50) * 0.05,
        ]))
        weights = get_sample_weights_by_returns(returns, span=10)

        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights > 0)

    def test_uniqueness_weights(self):
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        close_times = pd.Series(index=dates)
        for i, date in enumerate(dates):
            close_times.iloc[i] = date + timedelta(days=1 if i < 10 else 5)

        weights = get_sample_weights_by_uniqueness(close_times)

        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights > 0)
        assert weights.iloc[:10].mean() > weights.iloc[10:].mean()

    def test_concurrent_labels(self):
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        close_times = pd.Series([date + timedelta(days=3) for date in dates], index=dates)
        concurrent = get_num_concurrent_labels(close_times)

        assert np.all(concurrent > 0)
        assert concurrent.iloc[5] >= concurrent.iloc[0]


class TestSequentialBootstrap:
    """Test Sequential Bootstrap."""

    def test_bootstrap_size(self):
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )
        indices = seq_bootstrap(X, n_samples=50, random_state=42)

        assert len(indices) == 50
        assert np.all(indices >= 0)
        assert np.all(indices < len(X))

    def test_bootstrap_respects_weights(self):
        X = pd.DataFrame(
            np.random.randn(100, 5),
            index=pd.date_range('2020-01-01', periods=100, freq='D')
        )
        weights = pd.Series(0.01, index=X.index)
        weights.iloc[:10] = 10.0
        weights = weights / weights.sum()

        indices = seq_bootstrap(X, sample_weights=weights, n_samples=1000, random_state=42)
        first_10_count = np.sum(indices < 10)
        assert first_10_count > 100

    def test_bootstrap_reproducibility(self):
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
        series = pd.Series(np.log(np.arange(1, 101, dtype=float)))
        diff_series = frac_diff_ffd(series, d=0.5, threshold=0.01)

        assert len(diff_series) < len(series)
        assert diff_series.notna().sum() > 0

    def test_frac_diff_d_equals_zero(self):
        series = pd.Series(np.random.randn(100))
        diff_series = frac_diff_ffd(series, d=0.0, threshold=0.01)

        common_idx = diff_series.index
        assert np.allclose(
            diff_series.loc[common_idx].values,
            series.loc[common_idx].values,
            rtol=0.1
        )

    def test_frac_diff_d_equals_one(self):
        series = pd.Series(np.random.randn(100).cumsum())
        frac_diff = frac_diff_ffd(series, d=1.0, threshold=0.01)
        regular_diff = series.diff()

        common_idx = frac_diff.index
        correlation = np.corrcoef(
            frac_diff.loc[common_idx].values,
            regular_diff.loc[common_idx].dropna().values[:len(frac_diff)]
        )[0, 1]
        assert correlation > 0.9

    def test_frac_diff_stationarity(self):
        np.random.seed(42)
        series = pd.Series(np.random.randn(200).cumsum())
        diff_series = frac_diff_ffd(series, d=0.5, threshold=0.01)

        if len(diff_series) > 1:
            assert diff_series.std() != series.std()


class TestTripleBarrier:
    """Test triple-barrier labeling."""

    def test_triple_barrier_profit(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = pd.Series([100] + [100 + i*0.5 for i in range(1, 100)], index=dates)
        events = pd.DatetimeIndex([dates[0]])

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.03, stop_loss=-0.02,
            vertical_barrier_timedelta=pd.Timedelta(hours=50)
        )

        assert len(barriers) == 1
        assert barriers.iloc[0]['label'] == 1
        assert barriers.iloc[0]['barrier_touched'] == 'profit'

    def test_triple_barrier_stop_loss(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = pd.Series(np.linspace(100, 95, 100), index=dates)
        events = pd.DatetimeIndex([dates[0]])

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.03, stop_loss=-0.02,
            vertical_barrier_timedelta=pd.Timedelta(hours=50)
        )

        assert len(barriers) == 1
        assert barriers.iloc[0]['label'] == -1
        assert barriers.iloc[0]['barrier_touched'] == 'stop'

    def test_triple_barrier_vertical(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        prices = pd.Series(100 + np.random.randn(50) * 0.1, index=dates)
        events = pd.DatetimeIndex([dates[0]])

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.10, stop_loss=-0.10,
            vertical_barrier_timedelta=pd.Timedelta(hours=20)
        )

        assert len(barriers) == 1
        assert barriers.iloc[0]['barrier_touched'] == 'vertical'
        assert barriers.iloc[0]['label'] in [-1, 1]

    def test_triple_barrier_multiple_events(self):
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5), index=dates)
        events = pd.DatetimeIndex(dates[::20])

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.02, stop_loss=-0.02,
            vertical_barrier_timedelta=pd.Timedelta(hours=10)
        )

        assert len(barriers) >= len(events) * 0.8
        assert barriers['label'].isin([-1, 1]).all()
        assert barriers['barrier_touched'].isin(['profit', 'stop', 'vertical']).all()

    def test_triple_barrier_with_side(self):
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = pd.Series(np.linspace(100, 105, 100), index=dates)
        events = pd.DatetimeIndex([dates[0], dates[50]])
        side = pd.Series([1, -1], index=events)

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.03, stop_loss=-0.02,
            vertical_barrier_timedelta=pd.Timedelta(hours=30),
            side=side
        )

        assert len(barriers) == 2
        assert barriers.iloc[0]['label'] == 1
        assert barriers.iloc[1]['label'] == -1

    def test_bins_from_triple_barrier(self):
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106], index=dates)
        events = pd.DatetimeIndex([dates[0], dates[2], dates[4]])

        barriers = get_events_triple_barrier(
            close=prices, events=events,
            profit_target=0.02, stop_loss=-0.02,
            vertical_barrier_timedelta=pd.Timedelta(hours=3)
        )

        bins = get_bins_from_triple_barrier(barriers, prices)

        assert bins.isin([0, 1]).all()
        assert len(bins) == len(barriers)


class TestMetaLabeling:
    """Test Meta-Labeling."""

    def test_meta_labels_basic(self):
        events = pd.DataFrame({
            'target': [0.05, -0.03, 0.02, -0.01, 0.04],
            'side': [1, -1, 1, 1, -1],
        })
        predictions = pd.Series([1, -1, 1, 1, -1], index=events.index)
        meta_labels = get_meta_labels(events, predictions)

        assert meta_labels.iloc[0] == 1
        assert meta_labels.iloc[1] == 1
        assert meta_labels.iloc[2] == 1
        assert meta_labels.iloc[3] == 0
        assert meta_labels.iloc[4] == 0

    def test_meta_labels_all_correct(self):
        events = pd.DataFrame({'target': [0.05, -0.03, 0.02], 'side': [1, -1, 1]})
        predictions = pd.Series([1, -1, 1], index=events.index)
        meta_labels = get_meta_labels(events, predictions)

        assert np.all(meta_labels == 1)

    def test_meta_labels_all_wrong(self):
        events = pd.DataFrame({'target': [0.05, -0.03, 0.02], 'side': [-1, 1, -1]})
        predictions = pd.Series([-1, 1, -1], index=events.index)
        meta_labels = get_meta_labels(events, predictions)

        assert np.all(meta_labels == 0)


class TestOptimalFracDiff:
    """Test optimal fractional differentiation order finding."""

    def test_optimal_d_for_stationary_series(self):
        series = pd.Series(np.random.randn(200))
        optimal_d = get_optimal_frac_diff_order(series, max_d=1.0, step=0.1)
        assert optimal_d <= 0.3

    def test_optimal_d_for_nonstationary_series(self):
        series = pd.Series(np.random.randn(200).cumsum())
        optimal_d = get_optimal_frac_diff_order(series, max_d=1.0, step=0.1)
        assert optimal_d >= 0.3


class TestLopezDePradoEnsemble:
    """Test Lopez de Prado ensemble wrapper."""

    def test_ensemble_predict(self):
        from freqtrade.freqai.prediction_models.LightGBMClassifierLopezDePrado import (
            LopezDePradoEnsemble
        )
        from sklearn.ensemble import RandomForestClassifier

        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        models = []
        for _ in range(3):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            models.append(model)

        ensemble = LopezDePradoEnsemble(models)
        predictions = ensemble.predict(X_test)

        assert len(predictions) == len(X_test)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_ensemble_predict_proba(self):
        from freqtrade.freqai.prediction_models.LightGBMClassifierLopezDePrado import (
            LopezDePradoEnsemble
        )
        from sklearn.ensemble import RandomForestClassifier

        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        models = []
        for _ in range(3):
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            models.append(model)

        ensemble = LopezDePradoEnsemble(models)
        probas = ensemble.predict_proba(X_test)

        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)
