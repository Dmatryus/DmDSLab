"""Воспроизводимость прогонов: фиксация random_seed и детерминированная
сортировка данных по колонке-идентификатору.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы по ARCHITECTURE §3,
тела поднимают `NotImplementedError`.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["set_global_seed", "sort_by_id", "compute_params_hash"]


def set_global_seed(random_seed: int | None) -> None:
    """Фиксирует глобальное зерно случайности.

    Args:
        random_seed: Зерно случайности. ``None`` — порядок не фиксируется.
    """
    raise NotImplementedError


def sort_by_id(data: pd.DataFrame, id_column: str | None) -> pd.DataFrame:
    """Детерминированно сортирует таблицу по колонке-идентификатору строк.

    Args:
        data: Исходная таблица.
        id_column: Имя колонки-идентификатора. ``None`` — сортировка не
            выполняется, таблица возвращается как есть.

    Returns:
        Детерминированно упорядоченная таблица.
    """
    raise NotImplementedError


def compute_params_hash(params: dict[str, object]) -> str:
    """Вычисляет стабильный хэш параметров прогона.

    Args:
        params: Словарь параметров прогона.

    Returns:
        Хэш-строка, используемая как `run_id`.
    """
    raise NotImplementedError
