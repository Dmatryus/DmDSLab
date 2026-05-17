"""Локальный и глобальный лидерборд: чтение, запись, ранжирование.

Заглушка каркаса (E-001.4): сигнатуры зафиксированы по ARCHITECTURE §3/§5,
тела поднимают `NotImplementedError`.

`dataset_id: str` принимается как готовая строка-идентификатор записи
лидерборда. Механизм генерации `dataset_id` — открытый вопрос OQ-2,
отложен в эпик E-002; каркас от него не зависит.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from ..core.orchestrator import CVResult

__all__ = ["update", "finalize_ranking", "read_leaderboard"]


def update(
    conn: sqlite3.Connection,
    run_id: str,
    dataset_id: str,
    method_name: str,
    group: str,
    result: CVResult,
    is_baseline: bool = False,
) -> None:
    """Записывает результат метода в локальный и глобальный лидерборд.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.
        dataset_id: Идентификатор датасета (готовая строка).
        method_name: Имя FS-метода.
        group: Группа метода.
        result: Результат кросс-валидации метода.
        is_baseline: ``True``, если запись — пользовательский baseline.
    """
    raise NotImplementedError


def finalize_ranking(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Проставляет ранги результатам прогона и возвращает ранжированную
    таблицу.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона.

    Returns:
        Ранжированная таблица результатов прогона.
    """
    raise NotImplementedError


def read_leaderboard(
    conn: sqlite3.Connection,
    run_id: str | None = None,
    dataset_id: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Читает лидерборд из хранилища.

    Args:
        conn: Соединение с базой данных.
        run_id: Идентификатор прогона. ``None`` — глобальный лидерборд по
            всем прогонам.
        dataset_id: Идентификатор датасета для фильтрации. ``None`` — без
            фильтра по датасету.
        top_n: Число строк. ``None`` — полный список.

    Returns:
        Таблица лидерборда.
    """
    raise NotImplementedError
