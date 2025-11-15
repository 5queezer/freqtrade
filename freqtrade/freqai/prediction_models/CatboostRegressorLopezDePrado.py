import logging
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRegressor, Pool

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold
from freqtrade.freqai.lopez_de_prado_ensemble import LopezDePradoEnsemble


logger = logging.getLogger(__name__)


class CatboostRegressorLopezDePrado(BaseRegressionModel):
    """
    Lopez de Prado compliant CatBoost regressor using Purged K-Fold CV ensemble.

    Configuration example:
    "freqai": {
        "model_training_parameters": {
            "iterations": 100,
            "learning_rate": 0.05,
            "depth": 5
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
            logger.info("Training single CatBoost regressor (purged CV disabled or n_splits < 3)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk)

        logger.info(
            f"Training Lopez de Prado CatBoost regressor ensemble: {n_splits} purged folds, "
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

            train_pool = Pool(
                data=X_fold_train,
                label=y_fold_train,
                weight=weights_fold_train,
            )
            val_pool = Pool(
                data=X_fold_val,
                label=y_fold_val,
                weight=weights_fold_val,
            )

            model = CatBoostRegressor(
                allow_writing_files=True,
                train_dir=Path(dk.data_path) / f"fold_{fold_idx}",
                **self.model_training_parameters,
            )
            model.fit(X=train_pool, eval_set=val_pool)

            val_score = model.score(val_pool)
            fold_scores.append(val_score)
            models.append(model)

            logger.info(f"Fold {fold_idx}/{n_splits}: validation R² = {val_score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(
            f"CatBoost regressor ensemble complete: {len(models)} models, "
            f"avg R² = {avg_score:.4f} ± {std_score:.4f}"
        )
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk):
        train_data = Pool(
            data=X,
            label=y,
            weight=sample_weights,
        )
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            test_data = None
        else:
            test_data = Pool(
                data=data_dictionary["test_features"],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
            )

        init_model = self.get_init_model(dk.pair)

        model = CatBoostRegressor(
            allow_writing_files=True,
            train_dir=Path(dk.data_path),
            **self.model_training_parameters,
        )

        model.fit(X=train_data, eval_set=test_data, init_model=init_model)

        return model
