"""FS-методы группы ``shap`` — отбор по важностям на основе SHAP и
перестановок.

Сюда входят SHAP-importance, Permutation importance, Null importance
(ARCHITECTURE §2). SHAP используется только как метод отбора признаков,
не как инструмент объяснимости модели (CLAUDE NEVER §4).

Заглушка каркаса (E-001.4): классы-наследники `FSMethod` с финальными
метаданными; тела `fit_select` поднимают `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import FSMethod

__all__ = [
    "ShapImportanceMethod",
    "PermutationImportanceMethod",
    "NullImportanceMethod",
]


class ShapImportanceMethod(FSMethod):
    """Отбор признаков по SHAP-важностям."""

    name = "shap_importance"
    group = "shap"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class PermutationImportanceMethod(FSMethod):
    """Отбор признаков по важности перестановок (permutation importance)."""

    name = "permutation_importance"
    group = "permutation"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class NullImportanceMethod(FSMethod):
    """Отбор признаков по null-важностям (target permutation)."""

    name = "null_importance"
    group = "permutation"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError
