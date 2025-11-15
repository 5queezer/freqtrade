import logging
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold


logger = logging.getLogger(__name__)


class LightGBMClassifierLopezDePrado(BaseClassifierModel):
    """
    Lopez de Prado compliant classifier using Purged K-Fold CV ensemble.

    Demonstrates full Lopez de Prado workflow:
    - Purged K-Fold cross-validation with embargo
    - Ensemble training across multiple folds
    - Sample weighting (configured via feature_parameters)
    - Works with triple-barrier labels from strategy

    Configuration example:
    "freqai": {
        "model_training_parameters": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 5
        },
        "feature_parameters": {
            "use_purged_kfold_cv": true,
            "purged_cv_n_splits": 5,
            "purged_cv_embargo_pct": 0.01,
            "ldp_time_decay": 1.0,
            "ldp_sample_uniqueness": true,
            "label_horizon_candles": 20
        }
    }

    Uses ensemble averaging across all purged folds for robust predictions.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Train ensemble using Purged K-Fold cross-validation.

        If use_purged_kfold_cv is enabled, trains multiple models (one per fold)
        and ensembles predictions. Otherwise falls back to single model training.
        """
        feat_dict = self.freqai_info.get("feature_parameters", {})
        use_purged_cv = feat_dict.get("use_purged_kfold_cv", False)
        n_splits = feat_dict.get("purged_cv_n_splits", 5)
        embargo_pct = feat_dict.get("purged_cv_embargo_pct", 0.01)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        sample_weights = data_dictionary["train_weights"]

        if not use_purged_cv or n_splits < 3:
            logger.info("Training single model (purged CV disabled or n_splits < 3)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk)

        logger.info(
            f"Training Lopez de Prado ensemble: {n_splits} purged folds, "
            f"embargo={embargo_pct:.1%}"
        )

        # Build ensemble using purged CV
        close_times = None
        if "label_horizon_candles" in feat_dict and feat_dict["label_horizon_candles"] > 0:
            horizon = feat_dict["label_horizon_candles"]
            close_times = dk.train_dates + dk.train_dates.freq * horizon

        cv = PurgedKFold(
            n_splits=n_splits,
            samples_info_sets=close_times,
            pct_embargo=embargo_pct
        )

        models = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y[train_idx]
            weights_fold_train = sample_weights[train_idx]

            X_fold_val = X.iloc[val_idx]
            y_fold_val = y[val_idx]
            weights_fold_val = sample_weights[val_idx]

            model = LGBMClassifier(**self.model_training_parameters)
            model.fit(
                X=X_fold_train.to_numpy(),
                y=y_fold_train,
                sample_weight=weights_fold_train,
                eval_set=[(X_fold_val.to_numpy(), y_fold_val)],
                eval_sample_weight=[weights_fold_val],
            )

            val_score = model.score(
                X_fold_val.to_numpy(),
                y_fold_val,
                sample_weight=weights_fold_val
            )
            fold_scores.append(val_score)
            models.append(model)

            logger.info(f"Fold {fold_idx}/{n_splits}: validation score = {val_score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(
            f"Ensemble complete: {len(models)} models, "
            f"avg score = {avg_score:.4f} ± {std_score:.4f}"
        )

        # Return ensemble wrapper
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk):
        """Fallback to single model training (standard approach)."""
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            test_weights = None
        else:
            eval_set = [
                (
                    data_dictionary["test_features"].to_numpy(),
                    data_dictionary["test_labels"].to_numpy()[:, 0],
                )
            ]
            test_weights = data_dictionary["test_weights"]

        init_model = self.get_init_model(dk.pair)
        model = LGBMClassifier(**self.model_training_parameters)
        model.fit(
            X=X.to_numpy(),
            y=y,
            eval_set=eval_set,
            sample_weight=sample_weights,
            eval_sample_weight=[test_weights] if test_weights is not None else None,
            init_model=init_model,
        )
        return model


class LopezDePradoEnsemble:
    """
    Ensemble of models trained on different purged folds.
    Predictions are averaged across all models.
    """

    def __init__(self, models: list):
        self.models = models
        self.classes_ = models[0].classes_

    def predict(self, X):
        """Average predictions across ensemble."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def predict_proba(self, X):
        """Average probability predictions across ensemble."""
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
