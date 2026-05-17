"""Стабильные контракты данных публичного API.

Здесь зафиксированы датаклассы `Dataset` (входные данные бенчмарка) и
`BenchmarkResult` (результат прогона). Это контракты, на которые опираются
функциональные эпики E-004…E-009; их сигнатуры выверены по ARCHITECTURE §4.

Сами публичные функции (`run_benchmark` и др.) добавляются заглушками в
подзадаче E-001.4 — здесь только определения данных.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["Dataset", "BenchmarkResult"]


@dataclass
class Dataset:
    """Входной датасет бенчмарка.

    Хранится только в оперативной памяти и на диск никогда не пишется
    (ARCHITECTURE §5, §7).

    Attributes:
        data: Полная таблица — признаки вместе с целевой переменной.
        features_list: Список имён колонок-признаков.
        target: Имя целевой колонки.
        id: Имя колонки-идентификатора строки для детерминированной
            сортировки (обеспечивает воспроизводимость, критерий К2).
            ``None`` — сортировка по id не выполняется.
        support: Предотобранный пользователем набор признаков — точка
            сравнения в локальном лидерборде. ``None`` — точки сравнения нет.
    """

    data: pd.DataFrame
    features_list: list[str]
    target: str
    id: str | None = None
    support: list[str] | None = None


@dataclass
class BenchmarkResult:
    """Результат прогона бенчмарка.

    Attributes:
        run_id: Идентификатор прогона (хэш параметров).
        ranked_results: Ранжированная таблица результатов методов. Колонки:
            ``method_name``, ``group``, ``selected_features`` (list[str]),
            ``cv_score``, ``score_std``, ``rank``, ``duration_sec``.
        baseline_beaten: Был ли побит baseline. ``None`` — baseline не
            передавался.
    """

    run_id: str
    ranked_results: pd.DataFrame
    baseline_beaten: bool | None
