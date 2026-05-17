"""Общие утилиты для FS-методов.

> **Edit log:**
> - 2026-05-17 · v0.3.0 · execution-agent · создан (R-2: единая `is_classification`)

Модуль содержит каноническую реализацию определения типа задачи по
целевой переменной. До его появления каждая из четырёх групп методов
(`filter` / `wrapper` / `embedded` / `shap`) держала собственную копию
`_is_classification` с расходящейся семантикой (review E-005, R-2):
методы на одном датасете с ``task=None`` могли тренировать разные
прокси-модели. Эта функция сводит расхождения в один источник истины.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["is_classification"]


def is_classification(y: pd.Series) -> bool:
    """Эвристически определяет, является ли задача классификацией.

    Каноническая семантика (свод расхождений R-2):

    - нечисловой / категориальный dtype → классификация;
    - числовой с ``n_unique <= 2`` → классификация;
    - целочисленный (по значениям) с
      ``n_unique <= max(20, 0.05 * len(y))`` → классификация;
    - иначе → регрессия.

    Пропущенные значения отбрасываются (`dropna`) до анализа.

    Args:
        y: Целевая переменная.

    Returns:
        ``True`` для задачи классификации, ``False`` для регрессии.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return True
    values = y.dropna()
    n_unique = values.nunique()
    if n_unique <= 2:
        return True
    is_integer = bool(np.all(np.equal(np.mod(values.to_numpy(), 1), 0)))
    return is_integer and n_unique <= max(20, int(0.05 * len(values)))
