"""
Custom transformers for Lopez de Prado methods in FreqAI pipelines.

These transformers are compatible with sklearn pipelines and datasieve.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from freqtrade.freqai.lopez_de_prado import frac_diff_ffd


logger = logging.getLogger(__name__)


class FractionalDifferentiator(BaseEstimator, TransformerMixin):
    """
    Fractional Differentiation transformer for sklearn pipelines.

    From Lopez de Prado's "Advances in Financial Machine Learning", Chapter 5.

    Applies fractional differentiation to make time series stationary while
    preserving maximum memory (information content).

    Parameters
    ----------
    d : float, default=0.5
        Differentiation order. Typical range: 0.3 to 0.7
        - d=0: no differencing
        - d=0.5: half differentiation
        - d=1.0: full first difference
    threshold : float, default=0.01
        Minimum weight threshold for calculation
    columns : list, optional
        Specific columns to apply fractional differentiation.
        If None, applies to all numeric columns.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ('fracdiff', FractionalDifferentiator(d=0.4)),
    ...     ('scaler', MinMaxScaler())
    ... ])
    >>> X_transformed = pipe.fit_transform(X)
    """

    def __init__(
        self,
        d: float = 0.5,
        threshold: float = 0.01,
        columns: Optional[list] = None
    ):
        self.d = d
        self.threshold = threshold
        self.columns = columns
        self.feature_names_in_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer (learns column names).

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : ignored
            Not used, present for API compatibility

        Returns
        -------
        self : object
            Fitted transformer
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)

            # Determine which columns to differentiate
            if self.columns is None:
                # Apply to all numeric columns
                self.columns_to_diff_ = [
                    col for col in X.columns
                    if pd.api.types.is_numeric_dtype(X[col])
                ]
            else:
                self.columns_to_diff_ = self.columns
        else:
            raise ValueError("FractionalDifferentiator requires pandas DataFrame input")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fractional differentiation to features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            Fractionally differentiated features
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FractionalDifferentiator requires pandas DataFrame input")

        X_transformed = X.copy()

        # Apply fractional differentiation to selected columns
        for col in self.columns_to_diff_:
            if col in X_transformed.columns:
                try:
                    series = X_transformed[col]

                    # Apply fractional differentiation
                    diff_series = frac_diff_ffd(
                        series,
                        d=self.d,
                        threshold=self.threshold
                    )

                    # Handle NaN values at the beginning
                    # Fill with 0 or forward fill
                    X_transformed[col] = diff_series.reindex(X_transformed.index, fill_value=0)

                except Exception as e:
                    logger.warning(
                        f"Failed to apply fractional differentiation to column {col}: {e}. "
                        f"Keeping original values."
                    )

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if input_features is None:
            return self.feature_names_in_
        return input_features
