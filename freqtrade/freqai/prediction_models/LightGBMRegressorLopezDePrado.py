import logging
from typing import Any

from lightgbm import LGBMRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado_ensemble import LopezDePradoEnsemble, LopezDePradoMixin


logger = logging.getLogger(__name__)


class LightGBMRegressorLopezDePrado(BaseRegressionModel, LopezDePradoMixin):
    """
    Lopez de Prado compliant regressor using Purged K-Fold CV ensemble.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        config = self._get_ldp_config()
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weights = data_dictionary["train_weights"]

        if not self._should_use_ensemble(config):
            logger.info("Training single LightGBM regressor (purged CV disabled)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk)

        logger.info(
            f"Training ensemble: {config['n_splits']} purged folds, "
            f"embargo={config['embargo_pct']:.1%}"
        )

        cv = self._get_purged_cv(dk, config)
        models = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            w_train, w_val = sample_weights[train_idx], sample_weights[val_idx]

            model = LGBMRegressor(**self.model_training_parameters)
            model.fit(
                X=X_train, y=y_train, sample_weight=w_train,
                eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
            )

            val_score = model.score(X_val, y_val, sample_weight=w_val)
            fold_scores.append(val_score)
            models.append(model)
            logger.info(f"Fold {fold_idx}/{config['n_splits']}: R² = {val_score:.4f}")

        self._log_ensemble_complete(fold_scores, "LightGBM regressor")
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk):
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) == 0:
            eval_set, eval_weights = None, None
        else:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = data_dictionary["test_weights"]

        model = LGBMRegressor(**self.model_training_parameters)
        model.fit(
            X=X, y=y, eval_set=eval_set, sample_weight=sample_weights,
            eval_sample_weight=[eval_weights] if eval_weights is not None else None,
            init_model=self.get_init_model(dk.pair),
        )
        return model
