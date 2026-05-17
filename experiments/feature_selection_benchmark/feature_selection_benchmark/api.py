"""Точка входа публичного API и стабильные контракты данных.

Здесь зафиксированы датаклассы `Dataset` (входные данные бенчмарка) и
`BenchmarkResult` (результат прогона) — контракты, на которые опираются
функциональные эпики E-004…E-009; их сигнатуры выверены по ARCHITECTURE §4.

Здесь же объявлены публичные функции (`run_benchmark`, `get_leaderboard`,
`list_methods`, `register_method`) — на уровне каркаса это заглушки с
финальными сигнатурами и телами `raise NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .methods.base import FSMethod, MethodInfo

__all__ = [
    "Dataset",
    "BenchmarkResult",
    "run_benchmark",
    "get_leaderboard",
    "list_methods",
    "register_method",
]


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


def run_benchmark(
    dataset: Dataset,
    task: Literal["classification", "regression"] | None = None,
    methods: list[str] | None = None,
    cv: int = 5,
    n_trials: int = 20,
    checkpoint_dir: str | Path | None = None,
    random_seed: int | None = None,
    baseline_model: Any = None,
) -> BenchmarkResult:
    """Прогоняет FS-методы на датасете и возвращает ранжированный результат.

    Args:
        dataset: Входной датасет (признаки + целевая переменная).
        task: Тип задачи. ``None`` — автоопределение по типу и кардинальности
            целевой переменной.
        methods: Имена методов для прогона. ``None`` — все зарегистрированные
            методы, подходящие для задачи.
        cv: Число разбиений кросс-валидации.
        n_trials: Число итераций подбора гиперпараметров каждого метода.
        checkpoint_dir: Директория для промежуточных результатов. ``None`` —
            временная системная папка.
        random_seed: Зерно случайности для воспроизводимости. ``None`` — не
            фиксируется.
        baseline_model: Опциональная пользовательская модель — точка
            сравнения в лидерборде.

    Returns:
        Результат прогона с ранжированным лидербордом методов.
    """
    raise NotImplementedError


def get_leaderboard(
    run_id: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Возвращает лидерборд результатов.

    Args:
        run_id: Идентификатор прогона. ``None`` — глобальный лидерборд по
            всем прогонам.
        top_n: Число строк. ``None`` — полный список.

    Returns:
        Таблица лидерборда.
    """
    raise NotImplementedError


def list_methods(task: str | None = None) -> list[MethodInfo]:
    """Возвращает метаданные зарегистрированных FS-методов.

    Args:
        task: Тип задачи для фильтрации. ``None`` — все зарегистрированные
            методы.

    Returns:
        Список метаданных методов.
    """
    raise NotImplementedError


def register_method(
    name: str,
    method: FSMethod | type[FSMethod],
    group: str,
) -> None:
    """Регистрирует FS-метод в реестре.

    Args:
        name: Уникальное имя метода.
        method: Экземпляр или класс-наследник `FSMethod`.
        group: Группа метода — ``"filter"`` | ``"wrapper"`` |
            ``"embedded"`` | ``"shap"``.
    """
    raise NotImplementedError
