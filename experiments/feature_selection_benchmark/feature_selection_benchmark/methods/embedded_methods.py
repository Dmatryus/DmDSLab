"""FS-методы группы ``embedded`` — отбор встроен в обучение модели.

Сюда входят Lasso/ElasticNet, CatBoost.select_features, важности по
приросту в деревьях (ARCHITECTURE §2).

Заглушка каркаса (E-001.4): классы-наследники `FSMethod` с финальными
метаданными; тела `fit_select` поднимают `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import FSMethod

__all__ = [
    "LassoMethod",
    "CatBoostSelectMethod",
    "TreeGainImportanceMethod",
]


class LassoMethod(FSMethod):
    """Отбор признаков через L1-регуляризацию (Lasso / ElasticNet)."""

    name = "lasso"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class CatBoostSelectMethod(FSMethod):
    """Отбор признаков через ``CatBoost.select_features``."""

    name = "catboost_select"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class TreeGainImportanceMethod(FSMethod):
    """Отбор признаков по важности прироста в деревьях."""

    name = "tree_gain_importance"
    group = "embedded"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError
