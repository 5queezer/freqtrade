import logging
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado import PurgedKFold
from freqtrade.freqai.lopez_de_prado_ensemble import (
    LopezDePradoEnsemble,
    MultiTargetEnsembleWrapper,
)


logger = logging.getLogger(__name__)


class LightGBMClassifierMultiTargetLopezDePrado(BaseClassifierModel):
    """
    Lopez de Prado compliant LightGBM multi-target classifier.

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
            logger.info("Training multi-target LightGBM (purged CV disabled or n_splits < 3)")
            return self._train_multi_target_single(X, y, sample_weight, data_dictionary, dk)

        logger.info(
            f"Training multi-target LightGBM with Lopez de Prado: {n_splits} folds, "
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

                model = LGBMClassifier(**self.model_training_parameters)
                model.fit(
                    X=X_fold_train,
                    y=y_fold_train,
                    sample_weight=weights_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    eval_sample_weight=[weights_fold_val],
                )

                val_score = model.score(
                    X_fold_val, y_fold_val, sample_weight=weights_fold_val
                )
                fold_scores.append(val_score)
                models.append(model)

            avg_score = np.mean(fold_scores)
            logger.info(f"Target {target_idx + 1} ensemble: avg score = {avg_score:.4f}")
            target_ensembles.append(LopezDePradoEnsemble(models))

        return MultiTargetEnsembleWrapper(target_ensembles)

    def _train_multi_target_single(self, X, y, sample_weight, data_dictionary, dk):
        """Standard multi-target training without purged CV."""
        lgb = LGBMClassifier(**self.model_training_parameters)

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            eval_sets = [(None, None)] * data_dictionary["test_labels"].shape[1]  # type: ignore
            for i in range(data_dictionary["test_labels"].shape[1]):
                eval_sets[i] = (  # type: ignore
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i],
                )

        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {
                    "eval_set": eval_sets[i],
                    "eval_sample_weight": eval_weights,
                    "init_model": init_models[i],
                }
            )

        model = FreqaiMultiOutputClassifier(estimator=lgb)
        thread_training = self.freqai_info.get("multitarget_parallel_training", False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model
