import logging
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, Pool

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.lopez_de_prado_ensemble import LopezDePradoEnsemble, LopezDePradoMixin


logger = logging.getLogger(__name__)


class CatboostClassifierLopezDePrado(BaseClassifierModel, LopezDePradoMixin):
    """
    Lopez de Prado compliant CatBoost classifier using Purged K-Fold CV ensemble.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        config = self._get_ldp_config()
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weights = data_dictionary["train_weights"]

        if not self._should_use_ensemble(config):
            logger.info("Training single CatBoost classifier (purged CV disabled)")
            return self._train_single_model(X, y, sample_weights, data_dictionary, dk)

        logger.info(
            f"Training ensemble: {config['n_splits']} purged folds, "
            f"embargo={config['embargo_pct']:.1%}"
        )

        cv = self._get_purged_cv(dk, config)
        models = []
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            train_pool = Pool(
                data=X.iloc[train_idx], label=y.iloc[train_idx], weight=sample_weights[train_idx]
            )
            val_pool = Pool(
                data=X.iloc[val_idx], label=y.iloc[val_idx], weight=sample_weights[val_idx]
            )

            model = CatBoostClassifier(
                allow_writing_files=True, loss_function="MultiClass",
                train_dir=Path(dk.data_path) / f"fold_{fold_idx}",
                **self.model_training_parameters,
            )
            model.fit(X=train_pool, eval_set=val_pool)

            val_score = model.score(val_pool)
            fold_scores.append(val_score)
            models.append(model)
            logger.info(f"Fold {fold_idx}/{config['n_splits']}: score = {val_score:.4f}")

        self._log_ensemble_complete(fold_scores, "CatBoost classifier")
        return LopezDePradoEnsemble(models)

    def _train_single_model(self, X, y, sample_weights, data_dictionary, dk):
        train_data = Pool(data=X, label=y, weight=sample_weights)
        test_data = None
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            test_data = Pool(
                data=data_dictionary["test_features"],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
            )

        model = CatBoostClassifier(
            allow_writing_files=True, loss_function="MultiClass",
            train_dir=Path(dk.data_path), **self.model_training_parameters,
        )
        model.fit(X=train_data, eval_set=test_data, init_model=self.get_init_model(dk.pair))
        return model
