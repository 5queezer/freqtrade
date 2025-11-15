import logging
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor, Pool

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado_ensemble import (
    LopezDePradoEnsemble, LopezDePradoMixin, MultiTargetRegressorEnsembleWrapper,
)


logger = logging.getLogger(__name__)


class CatboostRegressorMultiTargetLopezDePrado(BaseRegressionModel, LopezDePradoMixin):
    """
    Lopez de Prado compliant CatBoost multi-target regressor.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        config = self._get_ldp_config()
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        if not self._should_use_ensemble(config):
            logger.info("Training multi-target CatBoost regressor (purged CV disabled)")
            return self._train_multi_target_single(X, y, sample_weight, data_dictionary, dk)

        logger.info(f"Training multi-target ensemble: {config['n_splits']} folds, {y.shape[1]} targets")

        cv = self._get_purged_cv(dk, config)
        target_ensembles = []

        for target_idx in range(y.shape[1]):
            y_single = y.iloc[:, target_idx]
            models, fold_scores = [], []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
                train_pool = Pool(X.iloc[train_idx], y_single.iloc[train_idx], sample_weight[train_idx])
                val_pool = Pool(X.iloc[val_idx], y_single.iloc[val_idx], sample_weight[val_idx])

                model = CatBoostRegressor(
                    allow_writing_files=True,
                    train_dir=Path(dk.data_path) / f"t{target_idx}_f{fold_idx}",
                    **self.model_training_parameters,
                )
                model.fit(X=train_pool, eval_set=val_pool)
                fold_scores.append(model.score(val_pool))
                models.append(model)

            logger.info(f"Target {target_idx + 1}: avg R² = {sum(fold_scores)/len(fold_scores):.4f}")
            target_ensembles.append(LopezDePradoEnsemble(models))

        return MultiTargetRegressorEnsembleWrapper(target_ensembles)

    def _train_multi_target_single(self, X, y, sample_weight, data_dictionary, dk):
        cbr = CatBoostRegressor(
            allow_writing_files=True, train_dir=Path(dk.data_path), **self.model_training_parameters
        )
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            for i in range(y.shape[1]):
                eval_sets[i] = Pool(
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i],
                    data_dictionary["test_weights"],
                )

        init_models = self.get_init_model(dk.pair).estimators_ if self.get_init_model(dk.pair) else [None] * y.shape[1]
        fit_params = [{"eval_set": eval_sets[i], "init_model": init_models[i]} for i in range(y.shape[1])]

        model = FreqaiMultiOutputRegressor(estimator=cbr)
        if self.freqai_info.get("multitarget_parallel_training", False):
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)
        return model
