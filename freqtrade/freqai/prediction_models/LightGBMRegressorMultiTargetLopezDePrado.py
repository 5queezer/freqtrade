import logging
from typing import Any

from lightgbm import LGBMRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado_ensemble import (
    LopezDePradoEnsemble, LopezDePradoMixin, MultiTargetRegressorEnsembleWrapper,
)


logger = logging.getLogger(__name__)


class LightGBMRegressorMultiTargetLopezDePrado(BaseRegressionModel, LopezDePradoMixin):
    """
    Lopez de Prado compliant LightGBM multi-target regressor.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        config = self._get_ldp_config()
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        if not self._should_use_ensemble(config):
            logger.info("Training multi-target LightGBM regressor (purged CV disabled)")
            return self._train_multi_target_single(X, y, sample_weight, data_dictionary, dk)

        logger.info(f"Training multi-target ensemble: {config['n_splits']} folds, {y.shape[1]} targets")

        cv = self._get_purged_cv(dk, config)
        target_ensembles = []

        for target_idx in range(y.shape[1]):
            y_single = y.iloc[:, target_idx]
            models, fold_scores = [], []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                model = LGBMRegressor(**self.model_training_parameters)
                model.fit(
                    X=X.iloc[train_idx], y=y_single.iloc[train_idx],
                    sample_weight=sample_weight[train_idx],
                    eval_set=[(X.iloc[val_idx], y_single.iloc[val_idx])],
                    eval_sample_weight=[sample_weight[val_idx]],
                )
                fold_scores.append(model.score(
                    X.iloc[val_idx], y_single.iloc[val_idx], sample_weight=sample_weight[val_idx]
                ))
                models.append(model)

            logger.info(f"Target {target_idx + 1}: avg R² = {sum(fold_scores)/len(fold_scores):.4f}")
            target_ensembles.append(LopezDePradoEnsemble(models))

        return MultiTargetRegressorEnsembleWrapper(target_ensembles)

    def _train_multi_target_single(self, X, y, sample_weight, data_dictionary, dk):
        lgb = LGBMRegressor(**self.model_training_parameters)
        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            for i in range(y.shape[1]):
                eval_sets[i] = [(data_dictionary["test_features"], data_dictionary["test_labels"].iloc[:, i])]

        init_models = self.get_init_model(dk.pair).estimators_ if self.get_init_model(dk.pair) else [None] * y.shape[1]
        fit_params = [{"eval_set": eval_sets[i], "eval_sample_weight": eval_weights, "init_model": init_models[i]} for i in range(y.shape[1])]

        model = FreqaiMultiOutputRegressor(estimator=lgb)
        if self.freqai_info.get("multitarget_parallel_training", False):
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)
        return model
