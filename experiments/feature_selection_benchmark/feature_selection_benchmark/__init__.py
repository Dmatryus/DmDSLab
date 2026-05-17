"""feature_selection_benchmark — бенчмарк методов отбора признаков.

Каркас пакета. Здесь реэкспортируется публичный API: функции
`run_benchmark`, `get_leaderboard`, `list_methods`, `register_method` и
контракты `Dataset`, `BenchmarkResult`, `FSMethod`, `MethodInfo`.
"""

from __future__ import annotations

from .api import (
    BenchmarkResult,
    Dataset,
    get_leaderboard,
    list_methods,
    register_method,
    run_benchmark,
)
from .methods.base import FSMethod, MethodInfo

__all__ = [
    "run_benchmark",
    "get_leaderboard",
    "list_methods",
    "register_method",
    "Dataset",
    "BenchmarkResult",
    "FSMethod",
    "MethodInfo",
]
