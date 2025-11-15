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
from freqtrade.freqai.lopez_de_prado_ensemble import LopezDePradoEnsemble, LopezDePradoMixin


logger = logging.getLogger(__name__)


class XGBoostClassifierLopezDePrado(BaseClassifierModel, LopezDePradoMixin):
    """
    Lopez de Prado compliant XGBoost classifier using Purged K-Fold CV ensemble.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        config = self._get_ldp_config()
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"].to_numpy()[:, 0]
        sample_weights = data_dictionary["train_weights"]

        le = LabelEncoder()
        if not is_integer_dtype(y):
            y = pd.Series(le.fit_transform(y), dtype="int64")

        if not self._should_use_ensemble(config):
            logger.info("Training single XGBoost classifier (purged CV disabled)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk, le)

        logger.info(
            f"Training ensemble: {config['n_splits']} purged folds, "
            f"embargo={config['embargo_pct']:.1%}"
        )

        cv = self._get_purged_cv(dk, config)
        models = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_train = X.iloc[train_idx].to_numpy()
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            X_val = X.iloc[val_idx].to_numpy()
            y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            w_train, w_val = sample_weights[train_idx], sample_weights[val_idx]

            model = XGBClassifier(**self.model_training_parameters)
            model.fit(X=X_train, y=y_train, sample_weight=w_train, eval_set=[(X_val, y_val)])

            val_score = model.score(X_val, y_val, sample_weight=w_val)
            fold_scores.append(val_score)
            models.append(model)
            logger.info(f"Fold {fold_idx}/{config['n_splits']}: score = {val_score:.4f}")

        self._log_ensemble_complete(fold_scores, "XGBoost classifier")
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk, le):
        eval_set = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            test_features = data_dictionary["test_features"].to_numpy()
            test_labels = data_dictionary["test_labels"].to_numpy()[:, 0]
            if not is_integer_dtype(test_labels):
                test_labels = pd.Series(le.transform(test_labels), dtype="int64")
            eval_set = [(test_features, test_labels)]

        model = XGBClassifier(**self.model_training_parameters)
        model.fit(
            X=X.to_numpy(), y=y, eval_set=eval_set, sample_weight=sample_weights,
            xgb_model=self.get_init_model(dk.pair)
        )
        return model

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs) -> tuple[DataFrame, np.ndarray]:
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
