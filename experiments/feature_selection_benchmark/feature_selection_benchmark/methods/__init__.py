"""methods — слой FS-методов: реестр, базовый класс и реализации методов.

Импорт этого пакета наполняет реестр built-in методами: каждый
группа-модуль импортируется здесь, его строки `register(...)`
выполняются как побочный эффект импорта (план E-005, A-1 —
саморегистрация). Поэтому `import feature_selection_benchmark` достаточно,
чтобы `list_registered` / `get_method` видели все методы.

Опциональные зависимости группа-модулей (`shap`, `boruta`, `catboost` и
др.) импортируются **лениво внутри методов**, поэтому импорт группа-модуля
не падает из-за отсутствующего пакета. Недоступный метод регистрируется и
помечается недоступным (см. `registry`, OQ-2).
"""

from __future__ import annotations

# Импорт группа-модулей (`embedded_methods` / `filter_methods` /
# `shap_methods` / `wrapper_methods`) ради побочного эффекта
# саморегистрации: их строки `register(...)` выполняются при импорте.
# Порядок импорта детерминирован (isort), наполнение реестра — тоже (К2).
from . import (
    embedded_methods,  # noqa: F401  (импорт ради регистрации)
    filter_methods,  # noqa: F401  (импорт ради регистрации)
    shap_methods,  # noqa: F401  (импорт ради регистрации)
    wrapper_methods,  # noqa: F401  (импорт ради регистрации)
)
from .base import FSMethod, HyperParam, MethodInfo
from .registry import (
    default_method_names,
    get_method,
    is_registered,
    list_registered,
    register,
)

__all__ = [
    "FSMethod",
    "HyperParam",
    "MethodInfo",
    "register",
    "get_method",
    "list_registered",
    "is_registered",
    "default_method_names",
]
