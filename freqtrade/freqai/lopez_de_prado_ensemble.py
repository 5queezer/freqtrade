"""
Ensemble wrappers for Lopez de Prado prediction models.

These classes wrap multiple models trained on purged CV folds
and average their predictions for more robust inference.
"""

import numpy as np


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
