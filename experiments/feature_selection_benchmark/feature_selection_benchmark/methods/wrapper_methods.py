"""FS-методы группы ``wrapper`` — отбор через перебор подмножеств с моделью.

Сюда входят RFE/RFECV, SequentialFeatureSelector, Boruta, BorutaShap,
Stability Selection (ARCHITECTURE §2).

Заглушка каркаса (E-001.4): классы-наследники `FSMethod` с финальными
метаданными; тела `fit_select` поднимают `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import FSMethod

__all__ = [
    "RFEMethod",
    "SequentialFeatureSelectorMethod",
    "BorutaMethod",
    "StabilitySelectionMethod",
]


class RFEMethod(FSMethod):
    """Рекурсивное исключение признаков (RFE/RFECV)."""

    name = "rfe"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class SequentialFeatureSelectorMethod(FSMethod):
    """Последовательный отбор признаков (forward/backward)."""

    name = "sequential_feature_selector"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class BorutaMethod(FSMethod):
    """Отбор признаков методом Boruta / BorutaShap."""

    name = "boruta"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class StabilitySelectionMethod(FSMethod):
    """Отбор признаков методом Stability Selection."""

    name = "stability_selection"
    group = "wrapper"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError
