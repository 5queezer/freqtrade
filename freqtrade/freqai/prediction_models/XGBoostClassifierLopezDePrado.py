import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold


logger = logging.getLogger(__name__)


class XGBoostClassifierLopezDePrado(BaseClassifierModel):
    """
    Lopez de Prado compliant XGBoost classifier using Purged K-Fold CV ensemble.

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
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        sample_weights = data_dictionary["train_weights"]

        # Handle label encoding for XGBoost
        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y), dtype="int64")

        if not use_purged_cv or n_splits < 3:
            logger.info("Training single XGBoost model (purged CV disabled or n_splits < 3)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk, le)

        logger.info(
            f"Training Lopez de Prado XGBoost ensemble: {n_splits} purged folds, "
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
            X_fold_train = X.iloc[train_idx].to_numpy()
            y_fold_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            weights_fold_train = sample_weights[train_idx]

            X_fold_val = X.iloc[val_idx].to_numpy()
            y_fold_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            weights_fold_val = sample_weights[val_idx]

            model = XGBClassifier(**self.model_training_parameters)
            model.fit(
                X=X_fold_train,
                y=y_fold_train,
                sample_weight=weights_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
            )

            val_score = model.score(X_fold_val, y_fold_val, sample_weight=weights_fold_val)
            fold_scores.append(val_score)
            models.append(model)

            logger.info(f"Fold {fold_idx}/{n_splits}: validation score = {val_score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(
            f"XGBoost ensemble complete: {len(models)} models, "
            f"avg score = {avg_score:.4f} ± {std_score:.4f}"
        )

        from freqtrade.freqai.prediction_models.LightGBMClassifierLopezDePrado import (
            LopezDePradoEnsemble
        )
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk, le):
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set = None
        else:
            test_features = data_dictionary["test_features"].to_numpy()
            test_labels = data_dictionary["test_labels"].to_numpy()[:, 0]

            if not is_integer_dtype(test_labels):
                test_labels = pd.Series(le.transform(test_labels), dtype="int64")

            eval_set = [(test_features, test_labels)]

        init_model = self.get_init_model(dk.pair)

        model = XGBClassifier(**self.model_training_parameters)
        model.fit(
            X=X.to_numpy(),
            y=y,
            eval_set=eval_set,
            sample_weight=sample_weights,
            xgb_model=init_model
        )

        return model

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, np.ndarray]:
        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        le = LabelEncoder()
        label = dk.label_list[0]
        labels_before = list(dk.data["labels_std"].keys())
        labels_after = le.fit_transform(labels_before).tolist()
        pred_df[label] = le.inverse_transform(pred_df[label])
        pred_df = pred_df.rename(
            columns={labels_after[i]: labels_before[i] for i in range(len(labels_before))}
        )

        return (pred_df, dk.do_predict)
