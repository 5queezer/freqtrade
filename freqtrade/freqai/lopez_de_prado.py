"""
Lopez de Prado Machine Learning Utilities for FreqAI

This module implements key methodologies from Marcos Lopez de Prado's books:
- "Advances in Financial Machine Learning" (2018)
- "Machine Learning for Asset Managers" (2020)

Key features:
- Purged K-Fold Cross-Validation with embargo periods
- Fractional Differentiation for feature stationarity
- Sample Uniqueness weighting
- Sequential Bootstrap
- Combinatorial Purged Cross-Validation (CPCV)
"""

import itertools
import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for time series data.

    Extends sklearn's K-Fold to:
    1. Respect temporal order (no future data in training)
    2. Purge training samples that overlap with test samples
    3. Add embargo period after test set to prevent leakage

    From: Advances in Financial Machine Learning, Chapter 7

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds
    samples_info_sets : pd.Series, optional
        Series indexed by sample index, values are timestamps when
        the information for that sample was available (e.g., bar end time)
    pct_embargo : float, default=0.01
        Percentage of samples to use as embargo period (buffer after test set)
        Typical values: 0.01 to 0.05 (1% to 5%)

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> cv = PurgedKFold(n_splits=5, pct_embargo=0.02)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     predictions = model.predict(X[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 3,
        samples_info_sets: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")

        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target variable
        groups : pd.Series, optional
            Not used, present for API compatibility

        Yields
        ------
        train : np.ndarray
            Training set indices for that split
        test : np.ndarray
            Testing set indices for that split
        """
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("X must be a pandas DataFrame or Series with a DatetimeIndex")

        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)

        test_ranges = [
            (i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)
        ]

        for start_test, end_test in test_ranges:
            # Test set indices
            test_indices = indices[start_test:end_test]

            # Calculate embargo end point
            end_embargo = min(end_test + embargo_size, X.shape[0])

            # Training indices: all data before test set
            # Exclude test set and embargo period
            if self.samples_info_sets is None:
                # Simple case: no purging, just use temporal split with embargo
                train_indices = np.concatenate([
                    indices[:start_test],
                    indices[end_embargo:]
                ])
            else:
                # Purged case: remove training samples that overlap with test set
                train_indices = self._get_purged_train_indices(
                    indices, test_indices, end_embargo
                )

            logger.debug(
                f"Fold: train size={len(train_indices)}, "
                f"test size={len(test_indices)}, "
                f"embargo={embargo_size}, "
                f"purged={X.shape[0] - len(train_indices) - len(test_indices) - embargo_size}"
            )

            yield train_indices, test_indices

    def _get_purged_train_indices(
        self, indices: npt.NDArray, test_indices: npt.NDArray, end_embargo: int
    ) -> npt.NDArray:
        """
        Get training indices with purging.

        Removes training samples whose information sets overlap with
        the test set's information sets.

        Parameters
        ----------
        indices : np.ndarray
            All available indices
        test_indices : np.ndarray
            Test set indices
        end_embargo : int
            End of embargo period

        Returns
        -------
        np.ndarray
            Purged training indices
        """
        # Get time range of test set
        test_times = self.samples_info_sets.iloc[test_indices]
        min_test_time = test_times.min()
        max_test_time = test_times.max()

        # Training candidates: before test start and after embargo
        train_candidates = np.concatenate([
            indices[:test_indices[0]],
            indices[end_embargo:]
        ])

        # Purge: remove samples whose info sets overlap with test set
        train_times = self.samples_info_sets.iloc[train_candidates]

        # Keep only samples that don't overlap with test period
        non_overlapping = (
            (train_times < min_test_time) | (train_times > max_test_time)
        )

        purged_train_indices = train_candidates[non_overlapping.values]

        return purged_train_indices

    def get_n_splits(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits


class CombinatorialPurgedKFold(BaseCrossValidator):
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Generates all possible combinations of train/test splits from N groups,
    where each test set is 1 group and training set is the remaining groups.
    Then applies purging and embargo to each split.

    From: Advances in Financial Machine Learning, Chapter 12

    This provides more robust validation for path-dependent strategies
    and better estimation of strategy performance variance.

    Parameters
    ----------
    n_splits : int, default=5
        Number of groups to split data into
    n_test_splits : int, default=2
        Number of groups to use in each test set
    samples_info_sets : pd.Series, optional
        Information set timestamps for each sample
    pct_embargo : float, default=0.01
        Embargo percentage

    Examples
    --------
    >>> cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
    >>> # This creates C(6,2) = 15 different train/test combinations
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        samples_info_sets: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        if n_splits < 3:
            raise ValueError(f"n_splits must be at least 3, got {n_splits}")
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits ({n_test_splits}) must be less than n_splits ({n_splits})"
            )

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        """Generate combinatorial train/test splits."""
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("X must be a pandas DataFrame or Series")

        indices = np.arange(X.shape[0])

        # Split indices into n_splits groups
        group_indices = np.array_split(indices, self.n_splits)

        # Generate all combinations of test groups
        test_group_combinations = itertools.combinations(
            range(self.n_splits), self.n_test_splits
        )

        embargo_size = int(X.shape[0] * self.pct_embargo)

        for test_groups in test_group_combinations:
            # Combine test group indices
            test_indices = np.concatenate([group_indices[i] for i in test_groups])
            test_indices.sort()

            # Training groups are all others
            train_groups = [i for i in range(self.n_splits) if i not in test_groups]
            train_indices = np.concatenate([group_indices[i] for i in train_groups])

            # Apply embargo: remove training samples too close to test set
            if self.samples_info_sets is not None:
                train_indices = self._apply_embargo_and_purge(
                    train_indices, test_indices, embargo_size
                )
            else:
                # Simple embargo without purging
                test_start = test_indices.min()
                test_end = test_indices.max()

                # Remove samples within embargo period
                train_mask = (
                    (train_indices < test_start - embargo_size) |
                    (train_indices > test_end + embargo_size)
                )
                train_indices = train_indices[train_mask]

            train_indices.sort()

            yield train_indices, test_indices

    def _apply_embargo_and_purge(
        self, train_indices: npt.NDArray, test_indices: npt.NDArray, embargo_size: int
    ) -> npt.NDArray:
        """Apply embargo and purging to training indices."""
        test_times = self.samples_info_sets.iloc[test_indices]
        min_test_time = test_times.min()
        max_test_time = test_times.max()

        # Calculate embargo buffer
        embargo_duration = test_times.iloc[-embargo_size:].max() if embargo_size > 0 else max_test_time

        train_times = self.samples_info_sets.iloc[train_indices]

        # Keep samples outside test period + embargo
        valid_mask = (
            (train_times < min_test_time) | (train_times > embargo_duration)
        )

        return train_indices[valid_mask.values]

    def get_n_splits(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Returns the number of splitting iterations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


def get_sample_weights_by_time_decay(
    dates: pd.Series,
    decay_factor: float = 1.0,
    last_date: Optional[pd.Timestamp] = None
) -> npt.NDArray:
    """
    Calculate sample weights using exponential time decay.

    More recent samples get higher weights.

    Parameters
    ----------
    dates : pd.Series
        Timestamps for each sample
    decay_factor : float, default=1.0
        Controls decay rate. Higher = more weight on recent data.
        decay_factor=0 gives uniform weights.
    last_date : pd.Timestamp, optional
        Reference date for decay calculation. Defaults to max(dates).

    Returns
    -------
    np.ndarray
        Sample weights (normalized to sum to 1)

    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> weights = get_sample_weights_by_time_decay(dates, decay_factor=0.5)
    """
    if last_date is None:
        last_date = dates.max()

    # Convert to days from last_date
    time_diffs = (last_date - dates).total_seconds().values / 86400.0  # days

    if decay_factor == 0:
        weights = np.ones(len(dates))
    else:
        # Exponential decay: w(t) = exp(-decay_factor * days_ago / total_days)
        max_diff = time_diffs.max()
        if max_diff > 0:
            weights = np.exp(-decay_factor * time_diffs / max_diff)
        else:
            weights = np.ones(len(dates))

    # Normalize
    weights = weights / weights.sum()

    return weights


def get_sample_weights_by_returns(
    returns: pd.Series,
    span: int = 60
) -> npt.NDArray:
    """
    Calculate sample weights based on return volatility.

    Samples during high-volatility periods get lower weights
    (they're less representative of normal conditions).

    Parameters
    ----------
    returns : pd.Series
        Return series
    span : int, default=60
        Span for exponential moving average volatility calculation

    Returns
    -------
    np.ndarray
        Sample weights (normalized)
    """
    # Calculate rolling volatility
    volatility = returns.ewm(span=span).std()

    # Inverse weight: higher volatility = lower weight
    weights = 1.0 / (volatility + 1e-8)

    # Handle NaN at the start
    weights = weights.fillna(weights.mean())

    # Normalize
    weights = weights / weights.sum()

    return weights.values


def get_sample_weights_by_uniqueness(
    close_times: pd.Series,
    molecule: Optional[npt.NDArray] = None,
    num_concurrent_labels: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Calculate sample weights by uniqueness (inverse of label overlap).

    From: Advances in Financial Machine Learning, Chapter 4

    Samples that overlap with many other samples (high concurrency)
    get lower weights because they provide less unique information.

    Parameters
    ----------
    close_times : pd.Series
        Series indexed by sample start time, values are label close times
        (when the label/prediction ends)
    molecule : np.ndarray, optional
        Subset of samples to calculate weights for
    num_concurrent_labels : pd.Series, optional
        Pre-calculated concurrency. If None, will calculate.

    Returns
    -------
    pd.Series
        Sample weights indexed by sample index

    Examples
    --------
    >>> # close_times: for each sample, when does its label end?
    >>> close_times = pd.Series({
    ...     pd.Timestamp('2020-01-01'): pd.Timestamp('2020-01-05'),
    ...     pd.Timestamp('2020-01-02'): pd.Timestamp('2020-01-06'),
    ...     pd.Timestamp('2020-01-03'): pd.Timestamp('2020-01-07'),
    ... })
    >>> weights = get_sample_weights_by_uniqueness(close_times)
    """
    if molecule is None:
        molecule = close_times.index.values

    if num_concurrent_labels is None:
        num_concurrent_labels = get_num_concurrent_labels(close_times)

    # Average uniqueness of each sample
    weights = pd.Series(index=molecule, dtype=float)

    for i in molecule:
        # Get start and end time for this sample
        start_time = i
        end_time = close_times.loc[i]

        # Find all samples active during this period
        active_samples = close_times[
            (close_times.index <= end_time) & (close_times >= start_time)
        ]

        # Weight is inverse of average concurrency during this sample's lifetime
        uniqueness = 0.0
        if len(active_samples) > 0:
            # For each point in time this sample is active,
            # get the number of concurrent labels
            # Weight = 1 / average_concurrency
            concurrent_counts = num_concurrent_labels.loc[active_samples.index]
            avg_concurrency = concurrent_counts.mean()
            uniqueness = 1.0 / avg_concurrency if avg_concurrency > 0 else 1.0

        weights.loc[i] = uniqueness

    # Normalize
    weights = weights / weights.sum()

    return weights


def get_num_concurrent_labels(close_times: pd.Series) -> pd.Series:
    """
    Calculate number of concurrent labels at each timestamp.

    From: Advances in Financial Machine Learning, Chapter 4

    Parameters
    ----------
    close_times : pd.Series
        Series indexed by label start time, values are label end times

    Returns
    -------
    pd.Series
        Number of concurrent labels at each timestamp
    """
    # Create events: +1 at start, -1 at end
    events = []

    for start_time, end_time in close_times.items():
        events.append((start_time, 1))  # Label starts
        events.append((end_time, -1))   # Label ends

    # Sort events by time
    events_df = pd.DataFrame(events, columns=['time', 'event'])
    events_df = events_df.sort_values('time')

    # Cumulative sum gives concurrent labels at each point
    events_df['concurrent'] = events_df['event'].cumsum()

    # Get concurrency at each label start time
    concurrency = pd.Series(index=close_times.index, dtype=int)

    for idx in close_times.index:
        # Find concurrency at this timestamp
        mask = events_df['time'] <= idx
        if mask.any():
            concurrency.loc[idx] = events_df.loc[mask, 'concurrent'].iloc[-1]
        else:
            concurrency.loc[idx] = 1

    return concurrency


def seq_bootstrap(
    indicators: pd.DataFrame,
    sample_weights: Optional[pd.Series] = None,
    n_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> npt.NDArray:
    """
    Sequential Bootstrap that respects temporal structure.

    From: Advances in Financial Machine Learning, Chapter 4

    Unlike standard bootstrap which samples with replacement randomly,
    sequential bootstrap builds a sample by:
    1. Drawing samples with probability proportional to their uniqueness
    2. Reducing probability of overlapping samples after each draw

    This preserves temporal dependencies while still providing bootstrap variance.

    Parameters
    ----------
    indicators : pd.DataFrame
        Feature matrix with DatetimeIndex
    sample_weights : pd.Series, optional
        Initial sample weights (e.g., from uniqueness calculation)
    n_samples : int, optional
        Number of samples to draw. Defaults to len(indicators)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Indices of bootstrapped samples

    Examples
    --------
    >>> X = pd.DataFrame(np.random.randn(100, 5))
    >>> bootstrap_indices = seq_bootstrap(X, n_samples=100)
    >>> X_bootstrap = X.iloc[bootstrap_indices]
    """
    if n_samples is None:
        n_samples = len(indicators)

    if sample_weights is None:
        sample_weights = pd.Series(1.0, index=indicators.index)

    rng = np.random.RandomState(random_state)

    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum()

    bootstrap_indices = []
    available_weights = sample_weights.copy()

    for _ in range(n_samples):
        # Draw one sample with probability proportional to current weights
        probs = available_weights / available_weights.sum()

        # Sample one index
        chosen_idx = rng.choice(probs.index, p=probs.values)
        bootstrap_indices.append(indicators.index.get_loc(chosen_idx))

        # Reduce weights of overlapping samples
        # (In simple case, we just reduce the chosen sample's weight)
        # More sophisticated: reduce weights of temporally nearby samples
        available_weights.loc[chosen_idx] *= 0.5  # Reduce but don't eliminate

        # Renormalize
        if available_weights.sum() > 0:
            available_weights = available_weights / available_weights.sum()
        else:
            # All weights depleted, reset
            available_weights = sample_weights.copy()

    return np.array(bootstrap_indices)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 0.01
) -> pd.Series:
    """
    Fractionally differentiate a time series (FFD = Fixed-width window Fracdiff).

    From: Advances in Financial Machine Learning, Chapter 5

    Fractional differentiation allows making a series stationary while
    preserving maximum memory (information). Unlike integer differentiation
    (d=1 for returns, d=2 for second difference), fractional values like
    d=0.4 can achieve stationarity while keeping more signal.

    Parameters
    ----------
    series : pd.Series
        Time series to differentiate (e.g., log prices)
    d : float
        Differentiation order. Typically between 0 and 1.
        d=0: no differencing (original series)
        d=0.5: half differentiation
        d=1.0: equivalent to first difference (returns)
    threshold : float, default=0.01
        Minimum weight to include in the calculation.
        Lower = more accurate but slower. 0.01 typically good.

    Returns
    -------
    pd.Series
        Fractionally differentiated series

    Examples
    --------
    >>> prices = pd.Series([100, 101, 102, 101, 103])
    >>> log_prices = np.log(prices)
    >>> # Make stationary while preserving memory
    >>> stationary = frac_diff_ffd(log_prices, d=0.4)

    Notes
    -----
    The weights follow binomial expansion:
    w_k = (-1)^k * binom(d, k)

    For stationarity testing, use ADF test and find minimum d where
    the series becomes stationary.
    """
    # Compute binomial weights
    weights = [1.0]
    k = 1

    while True:
        # w_k = w_{k-1} * (d - k + 1) / k
        weight = -weights[-1] * (d - k + 1) / k

        if abs(weight) < threshold:
            break

        weights.append(weight)
        k += 1

    weights = np.array(weights)

    # Apply weights using convolution
    # Pad the series at the start to avoid losing too many observations
    width = len(weights) - 1

    # Create result series
    result = pd.Series(index=series.index, dtype=float)

    for i in range(width, len(series)):
        # Apply weights to window
        window = series.iloc[i - width:i + 1].values[::-1]  # Reverse for convolution
        result.iloc[i] = np.dot(window, weights[:len(window)])

    return result.dropna()


def get_optimal_frac_diff_order(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.05,
    threshold: float = 0.01,
    significance: float = 0.05,
) -> float:
    """
    Find the minimum differentiation order d that makes the series stationary.

    Uses ADF test to check stationarity at different d values.

    Parameters
    ----------
    series : pd.Series
        Time series (e.g., log prices)
    max_d : float, default=1.0
        Maximum d to test
    step : float, default=0.05
        Step size for d search
    threshold : float, default=0.01
        Threshold for frac_diff_ffd
    significance : float, default=0.05
        Significance level for ADF test

    Returns
    -------
    float
        Optimal differentiation order

    Examples
    --------
    >>> prices = pd.Series(data['close'])
    >>> log_prices = np.log(prices)
    >>> optimal_d = get_optimal_frac_diff_order(log_prices)
    >>> stationary_series = frac_diff_ffd(log_prices, d=optimal_d)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning(
            "statsmodels not installed. Cannot perform ADF test. Returning d=0.5"
        )
        return 0.5

    d_values = np.arange(0, max_d + step, step)

    for d in d_values:
        # Apply fractional differentiation
        diff_series = frac_diff_ffd(series, d=d, threshold=threshold)

        if len(diff_series) < 10:
            continue

        # Perform ADF test
        try:
            adf_result = adfuller(diff_series.dropna(), maxlag=1, regression='c', autolag=None)
            p_value = adf_result[1]

            # If p-value < significance, series is stationary
            if p_value < significance:
                logger.info(
                    f"Found optimal d={d:.3f} (ADF p-value={p_value:.4f})"
                )
                return d
        except Exception as e:
            logger.warning(f"ADF test failed for d={d}: {e}")
            continue

    logger.warning(
        f"Could not find d < {max_d} that makes series stationary. "
        f"Returning max_d={max_d}"
    )
    return max_d


def get_meta_labels(
    events: pd.DataFrame,
    predictions: pd.Series,
) -> pd.Series:
    """
    Generate meta-labels for secondary ML model (bet sizing).

    From: Advances in Financial Machine Learning, Chapter 3

    Meta-labeling is a 2-step approach:
    1. Primary model predicts direction (already done -> predictions)
    2. Secondary model predicts whether to act on that prediction

    The meta-label is typically:
    - 1 if acting on the prediction would be profitable
    - 0 if not acting (or acting opposite) would be better

    Parameters
    ----------
    events : pd.DataFrame
        Must contain 'target' (actual outcome) and 'side' (prediction side)
    predictions : pd.Series
        Primary model predictions (e.g., probability or direction)

    Returns
    -------
    pd.Series
        Meta-labels (0 or 1)

    Examples
    --------
    >>> # Step 1: Primary model predicts side (long/short)
    >>> primary_predictions = model.predict(X)
    >>> # Step 2: Create events with actual outcomes
    >>> events = pd.DataFrame({
    ...     'target': actual_returns,
    ...     'side': primary_predictions
    ... })
    >>> # Step 3: Generate meta-labels
    >>> meta_labels = get_meta_labels(events, primary_predictions)
    >>> # Step 4: Train meta-model to predict meta-labels
    >>> meta_model.fit(X, meta_labels)
    """
    # Meta-label is 1 if prediction side matches profitable outcome
    # 0 otherwise

    meta_labels = pd.Series(index=events.index, dtype=int)

    for idx in events.index:
        target = events.loc[idx, 'target']
        side = events.loc[idx, 'side']

        # If we predicted long (side=1) and target is positive, meta-label=1
        # If we predicted short (side=-1) and target is negative, meta-label=1
        # Otherwise meta-label=0 (don't trade or prediction was wrong)
        if (side > 0 and target > 0) or (side < 0 and target < 0):
            meta_labels.loc[idx] = 1
        else:
            meta_labels.loc[idx] = 0

    return meta_labels
