"""
Ensemble wrappers for Lopez de Prado prediction models.

These classes wrap multiple models trained on purged CV folds
and average their predictions for more robust inference.
"""

import logging

import numpy as np
import pandas as pd

from freqtrade.freqai.lopez_de_prado import PurgedKFold


logger = logging.getLogger(__name__)


class LopezDePradoMixin:
    """
    Mixin providing common configuration and utilities for Lopez de Prado models.
    """

    def _get_ldp_config(self):
        """Extract Lopez de Prado configuration from freqai_info."""
        feat_dict = self.freqai_info.get("feature_parameters", {})
        return {
            "use_purged_cv": feat_dict.get("use_purged_kfold_cv", False),
            "n_splits": feat_dict.get("purged_cv_n_splits", 5),
            "embargo_pct": feat_dict.get("purged_cv_embargo_pct", 0.01),
            "label_horizon": feat_dict.get("label_horizon_candles", 0),
        }

    def _should_use_ensemble(self, config):
        """Check if ensemble training should be used."""
        return config["use_purged_cv"] and config["n_splits"] >= 3

    def _get_purged_cv(self, dk, config, X=None, data_dictionary=None):
        """Create a PurgedKFold cross-validator aligned to training rows."""
        data_dictionary = data_dictionary or getattr(dk, "data_dictionary", {}) or {}
        close_times = self._get_event_end_times(dk, data_dictionary, X)
        if close_times is None and config["label_horizon"] > 0:
            train_dates = self._get_train_dates(dk, data_dictionary, X)
            if len(train_dates) >= 2:
                freq = pd.to_timedelta(train_dates.diff().median())
                close_times = train_dates + freq * config["label_horizon"]
            else:
                close_times = train_dates
            close_times.index = pd.DatetimeIndex(train_dates)
        return PurgedKFold(
            n_splits=config["n_splits"],
            samples_info_sets=close_times,
            pct_embargo=config["embargo_pct"],
        )

    def _get_event_end_times(self, dk, data_dictionary, X=None):
        """Return explicit event end times (t1) aligned to training rows when available."""
        close_times = None
        for key in ("train_event_end_times", "event_end_times", "train_t1", "t1"):
            value = data_dictionary.get(key)
            if value is not None and len(value) > 0:
                close_times = value
                break

        if close_times is None:
            for attr in ("train_event_end_times", "event_end_times", "train_t1", "t1"):
                value = getattr(dk, attr, None)
                if value is not None and len(value) > 0:
                    close_times = value
                    break

        if close_times is None:
            return None

        close_times = pd.Series(close_times).copy()
        if X is not None and len(close_times) != len(X):
            raise ValueError(
                "Purged CV event_end_times must align with train_features: "
                f"got {len(close_times)} timestamps for {len(X)} rows"
            )

        if isinstance(close_times.index, pd.DatetimeIndex):
            train_dates = pd.Series(close_times.index)
        else:
            train_dates = self._get_train_dates(dk, data_dictionary, X)
            close_times.index = pd.DatetimeIndex(train_dates)

        event_ends = pd.to_datetime(close_times.reset_index(drop=True))
        event_starts = pd.to_datetime(pd.Series(train_dates).reset_index(drop=True))
        if (event_ends < event_starts).any():
            raise ValueError("Purged CV event_end_times must not be earlier than train_dates")

        return close_times

    def _get_train_dates(self, dk, data_dictionary, X=None):
        """Return chronological training timestamps aligned to training rows."""
        train_dates = data_dictionary.get("train_dates")
        if train_dates is None or len(train_dates) == 0:
            train_dates = dk.train_dates
        train_dates = pd.Series(train_dates).reset_index(drop=True)

        if X is not None and len(train_dates) != len(X):
            raise ValueError(
                "Purged CV train_dates must align with train_features: "
                f"got {len(train_dates)} dates for {len(X)} rows"
            )
        if len(train_dates) >= 2 and not train_dates.is_monotonic_increasing:
            raise ValueError(
                "Purged CV requires chronologically ordered train_dates. "
                "Disable shuffle_after_split or preserve timestamp order."
            )

        return train_dates

    def _log_ensemble_complete(self, fold_scores, model_type="model"):
        """Log ensemble training completion statistics."""
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(
            f"{model_type} ensemble complete: {len(fold_scores)} models, "
            f"avg score = {avg_score:.4f} ± {std_score:.4f}"
        )
        return avg_score, std_score


class LopezDePradoEnsemble:
    """
    Ensemble of models trained on different purged folds.

    Classifiers use probability averaging when available (or majority voting
    otherwise). Regressors preserve continuous predictions by averaging floats.
    """

    def __init__(self, models: list):
        self.models = models
        self.classes_ = self._collect_classes(models)
        self._label_encoder = None

    @staticmethod
    def _collect_classes(models):
        """Collect classifier labels in stable order across all fold models."""
        classes = []
        for model in models:
            if not hasattr(model, "classes_"):
                return None
            for label in model.classes_:
                if not any(label == known for known in classes):
                    classes.append(label)
        return np.asarray(classes)

    def predict(self, X):
        """Predict with classifier-safe voting or regressor averaging."""
        if self.classes_ is not None:
            if all(hasattr(model, "predict_proba") for model in self.models):
                proba = self.predict_proba(X)
                return self.classes_[np.argmax(proba, axis=1)]
            predictions = np.asarray([model.predict(X) for model in self.models])
            return np.apply_along_axis(self._majority_vote, 0, predictions)

        predictions = np.asarray([model.predict(X) for model in self.models], dtype=float)
        return np.mean(predictions, axis=0)

    @staticmethod
    def _majority_vote(values):
        """Return the most common label without assuming ordinal classes."""
        labels, counts = np.unique(values, return_counts=True)
        return labels[np.argmax(counts)]

    def predict_proba(self, X):
        """Average probability predictions with class-column alignment."""
        if self.classes_ is None:
            raise AttributeError("Regressor ensembles do not provide predict_proba")

        aligned_probas = []
        for model in self.models:
            model_proba = np.asarray(model.predict_proba(X), dtype=float)
            model_classes = getattr(model, "classes_", self.classes_)
            aligned = np.zeros((model_proba.shape[0], len(self.classes_)), dtype=float)
            for source_col, label in enumerate(model_classes):
                target_cols = np.where(self.classes_ == label)[0]
                if len(target_cols) == 1:
                    aligned[:, target_cols[0]] = model_proba[:, source_col]
            aligned_probas.append(aligned)

        return np.mean(aligned_probas, axis=0)


class MultiTargetEnsembleWrapper:
    """
    Wrapper for multi-target classifier ensemble predictions.

    Each target has its own ensemble of models.
    """

    def __init__(self, target_ensembles: list):
        self.target_ensembles = target_ensembles
        self.classes_ = target_ensembles[0].classes_

    def predict(self, X):
        """Predict all targets using their respective ensembles."""
        predictions = np.array([ensemble.predict(X) for ensemble in self.target_ensembles])
        return predictions.T

    def predict_proba(self, X):
        """Return probabilities for first target only (FreqAI compatibility)."""
        return self.target_ensembles[0].predict_proba(X)


class MultiTargetRegressorEnsembleWrapper:
    """
    Wrapper for multi-target regressor ensemble predictions.

    Each target has its own ensemble of models.
    """

    def __init__(self, target_ensembles: list):
        self.target_ensembles = target_ensembles

    def predict(self, X):
        """Predict all targets using their respective ensembles."""
        predictions = np.array([ensemble.predict(X) for ensemble in self.target_ensembles])
        return predictions.T
