"""FS-методы группы ``filter`` — отбор по статистикам признаков.

Сюда входят VarianceThreshold, Pearson/Spearman-корреляция, Mutual
Information, mRMR, IV/WoE и др. (ARCHITECTURE §2).

Заглушка каркаса (E-001.4): классы-наследники `FSMethod` с финальными
метаданными; тела `fit_select` поднимают `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import FSMethod

__all__ = [
    "VarianceThresholdMethod",
    "CorrelationMethod",
    "MutualInformationMethod",
    "MRMRMethod",
]


class VarianceThresholdMethod(FSMethod):
    """Отбор признаков по порогу дисперсии."""

    name = "variance_threshold"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class CorrelationMethod(FSMethod):
    """Отбор признаков по корреляции с целевой переменной (Pearson/Spearman)."""

    name = "correlation"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class MutualInformationMethod(FSMethod):
    """Отбор признаков по взаимной информации с целевой переменной."""

    name = "mutual_information"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError


class MRMRMethod(FSMethod):
    """Отбор признаков методом minimum-Redundancy-Maximum-Relevance."""

    name = "mrmr"
    group = "filter"
    supported_tasks = ["classification", "regression"]

    def fit_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **hyperparams: Any,
    ) -> list[str]:
        raise NotImplementedError
