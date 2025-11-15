"""
Ensemble wrappers for Lopez de Prado prediction models.

These classes wrap multiple models trained on purged CV folds
and average their predictions for more robust inference.
"""

import logging

import numpy as np

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

    def _get_purged_cv(self, dk, config):
        """Create PurgedKFold cross-validator."""
        close_times = None
        if config["label_horizon"] > 0:
            close_times = dk.train_dates + dk.train_dates.freq * config["label_horizon"]

        return PurgedKFold(
            n_splits=config["n_splits"],
            samples_info_sets=close_times,
            pct_embargo=config["embargo_pct"]
        )

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
    Predictions are averaged across all models.

    Used for single-target classification and regression.
    """

    def __init__(self, models: list):
        self.models = models
        self.classes_ = models[0].classes_ if hasattr(models[0], 'classes_') else None

    def predict(self, X):
        """Average predictions across ensemble."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def predict_proba(self, X):
        """Average probability predictions across ensemble."""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)


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
