import logging
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold
from freqtrade.freqai.tensorboard import TBCallback


logger = logging.getLogger(__name__)


class XGBoostRegressorMultiTargetLopezDePrado(BaseRegressionModel):
    """
    Lopez de Prado compliant XGBoost multi-target regressor.

    Trains separate models for each target label using purged K-Fold CV.
    Combines ensemble training with multi-target prediction.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        feat_dict = self.freqai_info.get("feature_parameters", {})
        use_purged_cv = feat_dict.get("use_purged_kfold_cv", False)
        n_splits = feat_dict.get("purged_cv_n_splits", 5)
        embargo_pct = feat_dict.get("purged_cv_embargo_pct", 0.01)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        if not use_purged_cv or n_splits < 3:
            logger.info("Training XGBoost multi-target regressor (purged CV disabled or n_splits < 3)")
            return self._train_single_model(X, y, sample_weight, data_dictionary, dk)

        logger.info(
            f"Training multi-target XGBoost regressor with Lopez de Prado: {n_splits} folds, "
            f"{y.shape[1]} targets"
        )

        close_times = None
        if "label_horizon_candles" in feat_dict and feat_dict["label_horizon_candles"] > 0:
            horizon = feat_dict["label_horizon_candles"]
            close_times = dk.train_dates + dk.train_dates.freq * horizon

        cv = PurgedKFold(
            n_splits=n_splits,
            samples_info_sets=close_times,
            pct_embargo=embargo_pct
        )

        # Create ensemble for each target
        target_ensembles = []

        for target_idx in range(y.shape[1]):
            logger.info(f"Training ensemble for target {target_idx + 1}/{y.shape[1]}")
            y_single = y.iloc[:, target_idx]

            models = []
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                X_fold_train = X.iloc[train_idx]
                y_fold_train = y_single.iloc[train_idx]
                weights_fold_train = sample_weight[train_idx]

                X_fold_val = X.iloc[val_idx]
                y_fold_val = y_single.iloc[val_idx]
                weights_fold_val = sample_weight[val_idx]

                model = XGBRegressor(**self.model_training_parameters)
                model.set_params(callbacks=[TBCallback(dk.data_path)])
                model.fit(
                    X=X_fold_train,
                    y=y_fold_train,
                    sample_weight=weights_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    sample_weight_eval_set=[weights_fold_val],
                )
                model.set_params(callbacks=[])

                val_score = model.score(X_fold_val, y_fold_val, sample_weight=weights_fold_val)
                fold_scores.append(val_score)
                models.append(model)

            avg_score = np.mean(fold_scores)
            logger.info(f"Target {target_idx + 1} ensemble: avg R² = {avg_score:.4f}")

            from freqtrade.freqai.prediction_models.LightGBMClassifierLopezDePrado import (
                LopezDePradoEnsemble
            )
            target_ensembles.append(LopezDePradoEnsemble(models))

        return MultiTargetRegressorEnsembleWrapper(target_ensembles)

    def _train_single_model(self, X, y, sample_weight, data_dictionary, dk):
        """Standard multi-target training without purged CV."""
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"]), (X, y)]
            eval_weights = [data_dictionary["test_weights"], sample_weight]

        xgb_model = self.get_init_model(dk.pair)

        model = XGBRegressor(**self.model_training_parameters)
        model.set_params(callbacks=[TBCallback(dk.data_path)])
        model.fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model,
        )
        model.set_params(callbacks=[])

        return model


class MultiTargetRegressorEnsembleWrapper:
    """Wrapper for multi-target regressor ensemble predictions."""

    def __init__(self, target_ensembles: list):
        self.target_ensembles = target_ensembles

    def predict(self, X):
        predictions = np.array([ensemble.predict(X) for ensemble in self.target_ensembles])
        return predictions.T
