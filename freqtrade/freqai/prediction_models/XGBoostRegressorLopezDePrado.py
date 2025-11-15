import logging
from typing import Any

import numpy as np
from xgboost import XGBRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold
from freqtrade.freqai.lopez_de_prado_ensemble import LopezDePradoEnsemble
from freqtrade.freqai.tensorboard import TBCallback


logger = logging.getLogger(__name__)


class XGBoostRegressorLopezDePrado(BaseRegressionModel):
    """
    Lopez de Prado compliant XGBoost regressor using Purged K-Fold CV ensemble.

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
            "label_horizon_candles": 20
        }
    }
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        feat_dict = self.freqai_info.get("feature_parameters", {})
        use_purged_cv = feat_dict.get("use_purged_kfold_cv", False)
        n_splits = feat_dict.get("purged_cv_n_splits", 5)
        embargo_pct = feat_dict.get("purged_cv_embargo_pct", 0.01)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weights = data_dictionary["train_weights"]

        if not use_purged_cv or n_splits < 3:
            logger.info("Training single XGBoost regressor (purged CV disabled or n_splits < 3)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk)

        logger.info(
            f"Training Lopez de Prado XGBoost regressor ensemble: {n_splits} purged folds, "
            f"embargo={embargo_pct:.1%}"
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

        models = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            weights_fold_train = sample_weights[train_idx]

            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            weights_fold_val = sample_weights[val_idx]

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

            logger.info(f"Fold {fold_idx}/{n_splits}: validation R² = {val_score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(
            f"XGBoost regressor ensemble complete: {len(models)} models, "
            f"avg R² = {avg_score:.4f} ± {std_score:.4f}"
        )
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk):
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"]), (X, y)]
            eval_weights = [data_dictionary["test_weights"], sample_weights]

        xgb_model = self.get_init_model(dk.pair)

        model = XGBRegressor(**self.model_training_parameters)
        model.set_params(callbacks=[TBCallback(dk.data_path)])
        model.fit(
            X=X,
            y=y,
            sample_weight=sample_weights,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=xgb_model,
        )
        model.set_params(callbacks=[])

        return model
